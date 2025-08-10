"""
RunPod Deployment Script for Sonic Video Generator
This script runs on RunPod and processes video generation requests
"""

import os
import sys
import json
import base64
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
        # Use GPU 0
        pipe = Sonic(0)
        print("Sonic pipeline initialized successfully")
    return pipe

def download_file_from_base64(base64_data, file_extension):
    """Save base64 data to a temporary file"""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension)
    decoded_data = base64.b64decode(base64_data)
    temp_file.write(decoded_data)
    temp_file.close()
    return temp_file.name

def upload_to_cloudflare(video_path, webhook_data):
    """Upload the generated video back to Cloudflare"""
    try:
        with open(video_path, 'rb') as video_file:
            video_data = video_file.read()
            video_base64 = base64.b64encode(video_data).decode('utf-8')
            
        # Send webhook with video data
        response = requests.post(webhook_data['url'], json={
            'job_id': webhook_data['job_id'],
            'status': 'completed',
            'video_base64': video_base64,
            'metadata': {
                'file_size': len(video_data),
                'path': video_path
            }
        }, headers={
            'Authorization': f"Bearer {webhook_data.get('api_key', '')}",
            'Content-Type': 'application/json'
        })
        
        return response.status_code == 200
    except Exception as e:
        print(f"Error uploading to Cloudflare: {str(e)}")
        return False

def handler(event):
    """
    RunPod handler function for Sonic video generation
    
    Expected input format:
    {
        "input": {
            "image_base64": "base64_encoded_image",
            "audio_base64": "base64_encoded_audio",
            "prompt": "optional_prompt",
            "dynamic_scale": 1.0,
            "crop": false,
            "webhook_url": "https://worker.domain.com/api/webhook/video-complete",
            "webhook_data": {
                "job_id": "job_123",
                "api_key": "optional_api_key"
            }
        }
    }
    """
    try:
        print(f"DEBUG: Received event: {json.dumps(event, indent=2)}")
        
        # Check if input exists
        if "input" not in event:
            raise Exception("No 'input' field in event")
            
        job_input = event["input"]
        print(f"DEBUG: job_input keys: {list(job_input.keys())}")
        
        # Check required fields
        if "image_base64" not in job_input:
            raise Exception("Missing 'image_base64' in input")
        if "audio_base64" not in job_input:
            raise Exception("Missing 'audio_base64' in input")
            
        print(f"DEBUG: image_base64 length: {len(job_input['image_base64'])}")
        print(f"DEBUG: audio_base64 length: {len(job_input['audio_base64'])}")
        
        # Initialize Sonic if not already done
        sonic_pipe = initialize_sonic()
        
        # Create temporary files for input
        image_path = download_file_from_base64(job_input["image_base64"], ".png")
        audio_path = download_file_from_base64(job_input["audio_base64"], ".mp3")
        
        # Prepare output path
        output_dir = "/tmp/sonic_outputs"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"output_{event.get('id', 'video')}.mp4")
        
        # Get generation parameters
        dynamic_scale = job_input.get("dynamic_scale", 1.0)
        crop = job_input.get("crop", False)
        prompt = job_input.get("prompt", "")
        
        print(f"Processing video generation:")
        print(f"  Image: {image_path}")
        print(f"  Audio: {audio_path}")
        print(f"  Output: {output_path}")
        print(f"  Dynamic Scale: {dynamic_scale}")
        print(f"  Crop: {crop}")
        
        # Process face detection and cropping
        face_info = sonic_pipe.preprocess(image_path, expand_ratio=0.5)
        print(f"Face detection result: {face_info}")
        
        if face_info['face_num'] <= 0:
            raise Exception("No face detected in the image")
        
        # Crop image if requested
        if crop and face_info['face_num'] > 0:
            crop_image_path = image_path + '.crop.png'
            sonic_pipe.crop_image(image_path, crop_image_path, face_info['crop_bbox'])
            image_path = crop_image_path
            print(f"Image cropped to: {crop_image_path}")
        
        # Generate video with reduced steps for faster processing
        print("Starting video generation...")
        sonic_pipe.process(
            image_path, 
            audio_path, 
            output_path,
            min_resolution=512,
            inference_steps=15,  # Reduced from 25 to 15 for faster processing
            dynamic_scale=dynamic_scale
        )
        print(f"Video generated successfully: {output_path}")
        
        # Upload result to Cloudflare if webhook URL provided
        if "webhook_url" in job_input:
            webhook_data = {
                'url': job_input['webhook_url'],
                'job_id': job_input.get('webhook_data', {}).get('job_id', event.get('id')),
                'api_key': job_input.get('webhook_data', {}).get('api_key', '')
            }
            
            upload_success = upload_to_cloudflare(output_path, webhook_data)
            if not upload_success:
                print("Warning: Failed to upload video to Cloudflare")
        
        # Clean up temporary files
        try:
            os.remove(image_path)
            os.remove(audio_path)
            if crop and os.path.exists(crop_image_path):
                os.remove(crop_image_path)
        except:
            pass
        
        # Return success with video path
        return {
            "status": "completed",
            "output_path": output_path,
            "face_detected": face_info['face_num'] > 0,
            "message": "Video generated successfully"
        }
        
    except Exception as e:
        print(f"Error in handler: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Send failure webhook if URL provided
        if "input" in event and "webhook_url" in event["input"]:
            try:
                webhook_data = event["input"].get("webhook_data", {})
                requests.post(event["input"]["webhook_url"], json={
                    'job_id': webhook_data.get('job_id', event.get('id')),
                    'status': 'failed',
                    'error': str(e)
                }, headers={
                    'Authorization': f"Bearer {webhook_data.get('api_key', '')}",
                    'Content-Type': 'application/json'
                })
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
