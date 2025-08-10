"""
RunPod Handler - URL-Based Version with Enhanced Logging
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
        response = requests.get(url, stream=True, timeout=120)
        response.raise_for_status()
        
        total_size = 0
        # Write file in chunks to handle large files
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                temp_file.write(chunk)
                total_size += len(chunk)
        
        temp_file.close()
        print(f"Downloaded file: {temp_file.name} ({total_size} bytes)")
        return temp_file.name
        
    except Exception as e:
        temp_file.close()
        if os.path.exists(temp_file.name):
            os.remove(temp_file.name)
        raise Exception(f"Failed to download {url}: {str(e)}")

def handler(event):
    """
    ENHANCED URL-based RunPod handler for Sonic video generation
    
    Supports both URL and base64 input for compatibility
    """
    try:
        print(f"=== HANDLER START ===")
        print(f"Event keys: {list(event.keys())}")
        
        if "input" not in event:
            raise Exception("No 'input' field in event")
            
        job_input = event["input"]
        print(f"Input keys: {list(job_input.keys())}")
        
        # Method 1: URL-based input (preferred for large files)
        if "image_url" in job_input and "audio_url" in job_input:
            print("✅ Using URL-based input method (FAST)")
            
            image_path = download_file_from_url(job_input["image_url"], ".png")
            audio_path = download_file_from_url(job_input["audio_url"], ".mp3")
            
        # Method 2: Base64 input (fallback for small files)
        elif "image_base64" in job_input and "audio_base64" in job_input:
            print("⚠️ Using base64 input method (SLOWER)")
            
            import base64
            
            print(f"Image base64 length: {len(job_input['image_base64'])}")
            print(f"Audio base64 length: {len(job_input['audio_base64'])}")
            
            # Create temporary files
            image_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            audio_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            
            # Decode base64 data
            image_data = base64.b64decode(job_input["image_base64"])
            audio_data = base64.b64decode(job_input["audio_base64"])
            
            image_temp.write(image_data)
            audio_temp.write(audio_data)
            image_temp.close()
            audio_temp.close()
            
            image_path = image_temp.name
            audio_path = audio_temp.name
            
        else:
            raise Exception("❌ Missing input data. Provide either image_url/audio_url or image_base64/audio_base64")
        
        print(f"📁 Files ready:")
        print(f"   Image: {image_path} ({os.path.getsize(image_path)} bytes)")
        print(f"   Audio: {audio_path} ({os.path.getsize(audio_path)} bytes)")
        
        # Initialize Sonic
        print("🎯 Initializing Sonic...")
        sonic_pipe = initialize_sonic()
        
        # Setup output
        output_dir = "/tmp/sonic_outputs"
        os.makedirs(output_dir, exist_ok=True)
        job_id = job_input.get('job_id', event.get('id', 'video'))
        output_path = os.path.join(output_dir, f"output_{job_id}.mp4")
        
        # Get parameters
        dynamic_scale = job_input.get("dynamic_scale", 1.0)
        crop = job_input.get("crop", False)
        
        print(f"🎬 Processing parameters:")
        print(f"   Output: {output_path}")
        print(f"   Dynamic Scale: {dynamic_scale}")
        print(f"   Crop: {crop}")
        
        # Face detection
        print("👤 Running face detection...")
        face_info = sonic_pipe.preprocess(image_path, expand_ratio=0.5)
        print(f"Face result: {face_info}")
        
        if face_info['face_num'] <= 0:
            raise Exception("❌ No face detected in the image")
        
        # Crop if needed
        if crop and face_info['face_num'] > 0:
            crop_image_path = image_path + '.crop.png'
            sonic_pipe.crop_image(image_path, crop_image_path, face_info['crop_bbox'])
            image_path = crop_image_path
            print(f"✂️ Image cropped: {crop_image_path}")
        
        # Generate video with optimized settings
        print("🚀 Starting video generation...")
        print("   This may take several minutes for longer audio...")
        
        sonic_pipe.process(
            image_path, 
            audio_path, 
            output_path,
            min_resolution=512,
            inference_steps=20,  # Balanced for quality vs speed
            dynamic_scale=dynamic_scale
        )
        
        final_size = os.path.getsize(output_path)
        print(f"✅ Video generated successfully!")
        print(f"   Output: {output_path}")
        print(f"   Size: {final_size} bytes")
        
        # Clean up temporary files
        try:
            os.remove(image_path)
            os.remove(audio_path)
            if crop and 'crop_image_path' in locals():
                os.remove(crop_image_path)
            print("🧹 Temporary files cleaned up")
        except Exception as cleanup_error:
            print(f"⚠️ Cleanup warning: {cleanup_error}")
        
        return {
            "status": "completed",
            "output_path": output_path,
            "face_detected": face_info['face_num'] > 0,
            "file_size": final_size,
            "message": "Video generated successfully"
        }
        
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            "status": "failed",
            "error": str(e)
        }

# RunPod serverless handler
print("🎬 Sonic Handler Ready - Supporting both URL and Base64 inputs")
runpod.serverless.start({
    "handler": handler
})
