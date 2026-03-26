#!/bin/bin/python3
"""
调试脚本 - 检查服务返回的图片数据
"""
import requests
import base64
import json
import os
import sys
import cv2
import numpy as np

def test_service_response(server_url: str, image_path: str, algorithm: str = "ela"):
    """测试服务返回的完整数据"""
    
    print(f"=" * 60)
    print(f"调试服务返回数据")
    print(f"=" * 60)
    print(f"服务地址: {server_url}")
    print(f"图片路径: {image_path}")
    print(f"算法: {algorithm}")
    
    # 读取图片
    if not os.path.exists(image_path):
        print(f"✗ 图片不存在: {image_path}")
        return
    
    with open(image_path, 'rb') as f:
        image_base64 = base64.b64encode(f.read()).decode('utf-8')
    
    print(f"图片大小 (base64): {len(image_base64):,} bytes")
    
    # 发送请求
    payload = {
        'image_base64': image_base64,
        'algorithm': algorithm
    }
    
    try:
        response = requests.post(
            f"{server_url}/tamper_detection/v1/tamper_detect_img",
            data=payload,
            timeout=60
        )
        
        print(f"\nHTTP Status: {response.status_code}")
        
        if response.status_code != 200:
            print(f"响应文本: {response.text[:500]}")
            return
        
        result = response.json()
        
        # 检查各字段
        print(f"\n【状态检查】")
        print(f"  status: {result.get('status')}")
        print(f"  is_tampered: {result.get('is_tampered')}")
        print(f"  confidence: {result.get('confidence')}")
        print(f"  processing_time: {result.get('processing_time')}")
        print(f"  algorithm: {result.get('algorithm')}")
        
        # 检查图片数据
        print(f"\n【图片数据检查】")
        
        mask_base64 = result.get('mask_base64')
        if mask_base64:
            mask_bytes = base64.b64decode(mask_base64)
            print(f"  ✅ mask_base64: 有数据 ({len(mask_bytes):,} bytes)")
            # 尝试解码
            mask = cv2.imdecode(np.frombuffer(mask_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                print(f"      解码成功: {mask.shape}, 非零像素: {np.count_nonzero(mask)}")
            else:
                print(f"      ❌ 解码失败")
        else:
            print(f"  ❌ mask_base64: 无数据 (None)")
        
        marked_image_base64 = result.get('marked_image_base64')
        if marked_image_base64:
            marked_bytes = base64.b64decode(marked_image_base64)
            print(f"  ✅ marked_image_base64: 有数据 ({len(marked_bytes):,} bytes)")
            # 尝试解码
            marked = cv2.imdecode(np.frombuffer(marked_bytes, np.uint8), cv2.IMREAD_COLOR)
            if marked is not None:
                print(f"      解码成功: {marked.shape}")
            else:
                print(f"      ❌ 解码失败")
        else:
            print(f"  ❌ marked_image_base64: 无数据 (None)")
        
        tamper_regions = result.get('tamper_regions')
        if tamper_regions:
            print(f"  ✅ tamper_regions: {len(tamper_regions)} 个区域")
            for i, region in enumerate(tamper_regions[:3]):
                print(f"      区域{i+1}: {region}")
        else:
            print(f"  ❌ tamper_regions: 无数据 (None)")
        
        # 保存调试图片
        if mask_base64:
            mask = cv2.imdecode(np.frombuffer(base64.b64decode(mask_base64), np.uint8), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                cv2.imwrite('/tmp/debug_mask.png', mask)
                print(f"\n  掩码已保存: /tmp/debug_mask.png")
        
        if marked_image_base64:
            marked = cv2.imdecode(np.frombuffer(base64.b64decode(marked_image_base64), np.uint8), cv2.IMREAD_COLOR)
            if marked is not None:
                cv2.imwrite('/tmp/debug_marked.jpg', marked)
                print(f"  标注图已保存: /tmp/debug_marked.jpg")
        
        return result
        
    except Exception as e:
        print(f"✗ 请求失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='调试服务返回数据')
    parser.add_argument('--server', type=str, default='http://localhost:8000')
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--algorithm', type=str, default='ela')
    
    args = parser.parse_args()
    test_service_response(args.server, args.image, args.algorithm)