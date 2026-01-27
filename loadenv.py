import os
from dotenv import load_dotenv
from typing import Optional

class Config:
    def __init__(self):
        load_dotenv()  # 加载环境变量
    
    def get_api_key(self) -> Optional[str]:
        return os.getenv('mp_API_KEY')
    
    def get_ip(self) -> Optional[str]:
        return os.getenv('local_HOST')
    
    def get_host(self) -> Optional[str]:
        return os.getenv('HOST')

    def get_port(self) -> Optional[int]:
        port = os.getenv('PORT')
        return int(port) if port and port.isdigit() else None

    def get_username(self) -> Optional[str]:
        return os.getenv('USERNAME')

    def get_password(self) -> Optional[str]:
        return os.getenv('PASSWORD')

    def get_base_dir(self) -> Optional[str]:
        return os.getenv('base_dir')
    
    def validate_config(self) -> bool:
        """检查必要的环境变量是否已设置"""
        required_vars = ['mp_API_KEY', 'local_HOST', 'HOST', 'PORT', 'USERNAME', 'PASSWORD', 'base_dir']
        return all(os.getenv(var) for var in required_vars)



# 使用示例
if __name__ == "__main__":
    config = Config()
    if config.validate_config():
        api_key = config.get_api_key()
        print(f"API Key: {api_key}")
        # 使用 api_key...
    else:
        print("请设置必要的环境变量")