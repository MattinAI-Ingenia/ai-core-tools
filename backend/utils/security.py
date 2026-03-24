import hmac
import hashlib
import os
from config import SECRET_KEY

def generate_signature(path: str, username: str) -> str:
    """
    Generate HMAC signature for a file path and username.
    
    Args:
        path: The relative file path (e.g. 'conversations/123/image.png')
        username: The username of the requester
        
    Returns:
        Hex string signature
    """
    # Normalize path to ensure consistency (forward slashes)
    path = path.replace('\\', '/')
    if path.startswith('/'):
        path = path[1:]
        
    message = f"{path}:{username}"
    signature = hmac.new(
        SECRET_KEY.encode('utf-8'),
        message.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    return signature

def verify_signature(path: str, username: str, signature: str) -> bool:
    """
    Verify the signature for a file path and username.
    
    Args:
        path: The relative file path
        username: The username claiming access
        signature: The provided signature
        
    Returns:
        True if valid, False otherwise
    """
    if not signature or not username:
        return False
        
    expected_signature = generate_signature(path, username)
    return hmac.compare_digest(expected_signature, signature)


def generate_signed_url(file_path: str, username: str, base_url: str = None) -> str:
    """
    Generate a signed URL for accessing a static file.
    
    Args:
        file_path: Relative path to the file within TMP_BASE_FOLDER
        username: Username for signature generation
        base_url: Optional base URL (defaults to empty for relative URLs)
        
    Returns:
        Signed URL string
    """
    # Normalize path
    file_path = file_path.replace('\\', '/')
    if file_path.startswith('/'):
        file_path = file_path[1:]
    
    signature = generate_signature(file_path, username)
    
    base = base_url or ''
    return f"{base}/static/{file_path}?user={username}&sig={signature}"
