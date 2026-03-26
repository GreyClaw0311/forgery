#!/usr/bin/env python3
"""
调试脚本 - 检查服务返回格式
"""
import requests
import base64
import json
import os
import sys

def debug_service(server_url: str, image_path: str, algorithm: str = "ela"):
    """调试服务返回格式"""
    print(f"=" * 60)
    print(f"调试服务返回格式")
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
    
    print(f"图片大小: {len(image_base64)} bytes (base64)")
    
    # 构建请求
    payload = {
        'image_base64': image_base64,
        'algorithm': algorithm
    }
    
    print(f"\n发送请求...")
    
    try:
        response = requests.post(
            f"{server_url}/tamper_detection/v1/tamper_detect_img",
            data=payload,
            timeout=60
        )
        
        print(f"\nHTTP Status: {response.status_code}")
        print(f"HTTP Headers: {dict(response.headers)}")
        
        # 解析响应
        try:
            result = response.json()
            print(f"\n完整响应 (JSON):")
            print(json.dumps(result, indent=2, ensure_ascii=False))
            
            # 检查关键字段
            print(f"\n" + "=" * 60)
            print(f"关键字段检查:")
            print(f"=" * 60)
            
            status = result.get('status', '')
            print(f"  status: '{status}'")
            print(f"  status type: {type(status)}")
            print(f"  status.startswith('0001'): {status.startswith('0001')}")
            print(f"  status == '0001': {status == '0001'}")
            
            is_tampered = result.get('is_tampered')
            print(f"\n  is_tampered: {is_tampered} (type: {type(is_tampered)})")
            
            confidence = result.get('confidence')
            print(f"  confidence: {confidence} (type: {type(confidence)})")
            
            processing_time = result.get('processing_time')
            print(f"  processing_time: {processing_time} (type: {type(processing_time)})")
            
            algorithm = result.get('algorithm')
            print(f"  algorithm: {algorithm}")
            
            mask_base64 = result.get('mask_base64')
            print(f"  mask_base64: {'有数据' if mask_base64 else '无'}")
            
            # 检查是否有 data 嵌套
            data = result.get('data')
            if data:
                print(f"\n  发现 'data' 嵌套字段:")
                print(f"    data.is_tampered: {data.get('is_tampered')}")
                print(f"    data.confidence: {data.get('confidence')}")
            
            # 测试脚本判断逻辑
            print(f"\n" + "=" * 60)
            print(f"测试脚本判断逻辑:")
            print(f"=" * 60)
            if status.startswith('0001'):
                print(f"  ✅ 条件满足: status.startswith('0001') == True")
                print(f"  进入成功处理分支")
                pred_label = 1 if is_tampered else 0
                print(f"  pred_label = {pred_label}")
            else:
                print(f"  ❌ 条件不满足: status.startswith('0001') == False")
                print(f"  进入错误处理分支")
                print(f"  pred_label = -1")
                print(f"  error = '{result.get('message', 'Unknown error')}'")
            
        except json.JSONDecodeError as e:
            print(f"✗ JSON 解析失败: {e}")
            print(f"原始响应文本:")
            print(response.text[:1000])
        
    except requests.exceptions.ConnectionError as e:
        print(f"✗ 连接失败: {e}")
        print(f"请确保服务正在运行: python server_forgrey.py")
    except Exception as e:
        print(f"✗ 请求失败: {e}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='调试服务返回格式')
    parser.add_argument('--server', type=str, default='http://localhost:8000', help='服务地址')
    parser.add_argument('--image', type=str, required=True, help='测试图片路径')
    parser.add_argument('--algorithm', type=str, default='ela', help='算法')
    
    args = parser.parse_args()
    debug_service(args.server, args.image, args.algorithm)


if __name__ == '__main__':
    main()