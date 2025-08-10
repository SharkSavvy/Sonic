"""
RunPod Handler - URL-Based Version
Downloads files from URLs instead of receiving base64 data
"""

import os
import sys
import json
import tempfile
import requests
from pathlib import Path
import torch
import runpod

# Add Sonic to path
sys.path.append('/workspace/Sonic')

from sonic import Sonic

# Initialize Sonic pipeline
pipe = None

def initialize_sonic():
    """Initialize the Sonic pipeline once"""
    global pipe
    if pipe is None:
        print("Initializing Sonic pipeline...")
        pipe = Sonic(0)
        print("Sonic pipeline initialized successfully")
    return pipe

def download_file_from_url(url, file_extension):
    """Download file from URL to temporary file"""
    print(f"Downloading file from URL: {url}")
    
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension)
    
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        
        # Write file in chunks to handle large files
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                temp_file.write(chunk)
        
        temp_file.close()
        print(f"Downloaded file: {temp_file.name} ({os.path.getsize(temp_file.name)} bytes)")
        return temp_file.name
        
    except Exception as e:
        temp_file.close()
        if os.path.exists(temp_file.name):
            os.remove(temp_file.name)
        raise Exception(f"Failed to download {url}: {str(e)}")

def upload_video_to_webhook(video_path, webhook_url, job_id):
    """Upload completed video via webhook"""
    try:
        # For large files, we should upload to a temporary storage and send URL
        # For now, let's create a simple response
        file_size = os.path.getsize(video_path)
        print(f"Video completed: {video_path} ({file_size} bytes)")
        
        # In production, upload to temporary storage and send URL
        webhook_data = {
            'job_id': job_id,
            'status': 'completed',
            'video_url': f'https://temporary-storage.com/videos/{job_id}.mp4',
            'file_size': file_size,
            'message': 'Video generated successfully'
        }
        
        response = requests.post(webhook_url, json=webhook_data, timeout=30)
        print(f"Webhook response: {response.status_code}")
        return response.status_code == 200
        
    except Exception as e:
        print(f"Webhook error: {str(e)}")
        return False

def handler(event):
    """
    URL-based RunPod handler for Sonic video generation
    
    Expected input format:
    {
        "input": {
            "image_url": "https://storage.com/image.jpg",
            "audio_url": "https://storage.com/audio.mp3", 
            "dynamic_scale": 1.0,
            "crop": false,
            "webhook_url": "https://worker.domain.com/api/webhook/video-complete",
            "job_id": "job_123"
        }
    }
    """
    try:
        print(f"DEBUG: Received event: {json.dumps(event, indent=2)}")
        
        if "input" not in event:
            raise Exception("No 'input' field in event")
            
        job_input = event["input"]
        print(f"DEBUG: job_input keys: {list(job_input.keys())}")
        
        # Check for URL-based input first
        if "image_url" in job_input and "audio_url" in job_input:
            print("Using URL-based input method")
            
            # Download files from URLs
            image_path = download_file_from_url(job_input["image_url"], ".png")
            audio_path = download_file_from_url(job_input["audio_url"], ".mp3")
            
        elif "image_base64" in job_input and "audio_base64" in job_input:
            print("Using base64 input method (fallback)")
            
            # Fallback to base64 method
            import base64
            
            # Create temporary files for input
            image_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            audio_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            
            image_data = base64.b64decode(job_input["image_base64"])
            audio_data = base64.b64decode(job_input["audio_base64"])
            
            image_temp.write(image_data)
            audio_temp.write(audio_data)
            
            image_temp.close()
            audio_temp.close()
            
            image_path = image_temp.name
            audio_path = audio_temp.name
            
        else:
            raise Exception("Missing both URL and base64 inputs. Provide either image_url/audio_url or image_base64/audio_base64")
        
        print(f"Input files ready:")
        print(f"  Image: {image_path} ({os.path.getsize(image_path)} bytes)")
        print(f"  Audio: {audio_path} ({os.path.getsize(audio_path)} bytes)")
        
        # Initialize Sonic
        sonic_pipe = initialize_sonic()
        
        # Prepare output path
        output_dir = "/tmp/sonic_outputs"
        os.makedirs(output_dir, exist_ok=True)
        job_id = job_input.get('job_id', event.get('id', 'video'))
        output_path = os.path.join(output_dir, f"output_{job_id}.mp4")
        
        # Get generation parameters
        dynamic_scale = job_input.get("dynamic_scale", 1.0)
        crop = job_input.get("crop", False)
        
        print(f"Processing video generation:")
        print(f"  Output: {output_path}")
        print(f"  Dynamic Scale: {dynamic_scale}")
        print(f"  Crop: {crop}")
        
        # Process face detection
        face_info = sonic_pipe.preprocess(image_path, expand_ratio=0.5)
        print(f"Face detection result: {face_info}")
        
        if face_info['face_num'] <= 0:
            raise Exception("No face detected in the image")
        
        # Crop if requested
        if crop and face_info['face_num'] > 0:
            crop_image_path = image_path + '.crop.png'
            sonic_pipe.crop_image(image_path, crop_image_path, face_info['crop_bbox'])
            image_path = crop_image_path
            print(f"Image cropped to: {crop_image_path}")
        
        # Generate video with optimized settings for longer videos
        print("Starting video generation...")
        sonic_pipe.process(
            image_path, 
            audio_path, 
            output_path,
            min_resolution=512,
            inference_steps=20,  # Balanced quality vs speed
            dynamic_scale=dynamic_scale
        )
        print(f"Video generated successfully: {output_path}")
        
        # Send webhook if provided
        if "webhook_url" in job_input:
            webhook_success = upload_video_to_webhook(
                output_path, 
                job_input["webhook_url"], 
                job_id
            )
            if not webhook_success:
                print("Warning: Webhook delivery failed")
        
        # Clean up temporary files
        try:
            os.remove(image_path)
            os.remove(audio_path)
            if crop and 'crop_image_path' in locals():
                os.remove(crop_image_path)
        except:
            pass
        
        return {
            "status": "completed",
            "output_path": output_path,
            "face_detected": face_info['face_num'] > 0,
            "file_size": os.path.getsize(output_path),
            "message": "Video generated successfully"
        }
        
    except Exception as e:
        print(f"Error in handler: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Send failure webhook
        if "input" in event and "webhook_url" in event["input"]:
            try:
                job_id = event["input"].get("job_id", event.get("id"))
                requests.post(event["input"]["webhook_url"], json={
                    'job_id': job_id,
                    'status': 'failed',
                    'error': str(e)
                }, timeout=30)
            except:
                pass
        
        return {
            "status": "failed",
            "error": str(e)
        }

# RunPod serverless handler
runpod.serverless.start({
    "handler": handler
})
