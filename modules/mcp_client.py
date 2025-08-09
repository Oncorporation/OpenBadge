from gradio_client import Client
import requests
import json
import tempfile
import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any, Union, List
from PIL import Image
import logging
import time
import io
import base64

if __name__ == "__main__":
    parent_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(parent_dir))
# Import constants for badge generation
from modules.constants import badge_negative_prompt, HF_API_TOKEN, STYLE_TEMPLATES

# Configure logging for better error tracking
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MCPImageGenerator:
    """
    A repeatable function class for calling various MCP image generation servers.
    Uses proper MCP protocol with Server-Sent Events (SSE) transport.
    """
    
    def __init__(self):
        """Initialize the MCP Image Generator with default server configurations."""
        self.servers = {
            "flux_lora_dlc": {
                "mcp_url": "https://prithivmlmods-flux-lora-dlc.hf.space/gradio_api/mcp/",
                "description": "FLUX LoRA DLC - 260+ artistic styles for varied illustration badge designs"
            },
            "flux_schnell": {
                "mcp_url": "https://evalstate-flux1-schnell.hf.space/gradio_api/mcp/",
                "description": "FLUX SCHNELL - Fast Professional, realistic illustration badge designs"
            },
            "qwen_image_diffusion": {
                "mcp_url": "https://prithivmlmods-qwen-image-diffusion.hf.space/gradio_api/mcp/",
                "description": "Qwen Image Diffusion - Advanced MCP integration for illustration badge generation"
            }
        }
        self.current_server = None
        # Cache for tool schemas to avoid repeated requests
        self._tool_schemas_cache = {}
    
    def _call_mcp_server(self, mcp_url: str, tool_name: str, arguments: Dict[str, Any], hf_token: str = None) -> Optional[Any]:
        """
        Call an MCP server using Server-Sent Events (SSE) transport with Hugging Face authentication.
        
        Args:
            mcp_url (str): The MCP server SSE endpoint URL (ending with /mcp/)
            tool_name (str): Name of the tool to call on the MCP server
            arguments (Dict[str, Any]): Arguments to send to the tool
            hf_token (str, optional): Hugging Face access token for authentication. 
                                    If None, tries to use HF_API_TOKEN from constants.
            
        Returns:
            Optional[Any]: Response from MCP server or None if failed
        """
        try:
            # Import HF_API_TOKEN from constants if no token provided
            if not hf_token:
                from modules.constants import HF_API_TOKEN
                hf_token = HF_API_TOKEN
                if not hf_token:
                    logger.warning("No Hugging Face access token provided or found in constants. Proceeding without authentication.")
            
            # Prepare MCP request payload for tools/call
            mcp_request = {
                "jsonrpc": "2.0",
                "id": f"req_{int(time.time())}",
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": arguments
                }
            }
            
            logger.info(f"Calling MCP server at {mcp_url} with tool {tool_name}")
            logger.debug(f"MCP request: {json.dumps(mcp_request, indent=2)}")
            
            # Prepare headers with optional authentication
            headers = {
                "Accept": "application/json, text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "application/json"
            }
            
            # Add authentication header if token is available
            if hf_token:
                headers["Authorization"] = f"Bearer {hf_token}"
                logger.debug("Added Hugging Face authentication to request headers")
            
            # Make SSE request to MCP server
            response = requests.post(
                mcp_url,
                json=mcp_request,
                headers=headers,
                stream=True,
                timeout=120  # 2 minute timeout for image generation
            )
            
            if response.status_code != 200:
                logger.error(f"MCP server returned status code: {response.status_code}")
                logger.error(f"Response: {response.text}")
                
                # Check for authentication-related errors
                if response.status_code == 401:
                    logger.error("Authentication failed - check your HF_TOKEN")
                elif response.status_code == 403:
                    logger.error("Access forbidden - your HF_TOKEN may not have sufficient permissions")
                elif response.status_code == 429:
                    logger.error("Rate limit exceeded - try again later or use authentication")
                
                return None
            
            # Parse SSE response stream
            result = None
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                    
                # Handle SSE event format
                if line.startswith("event: "):
                    event_type = line[7:].strip()
                    logger.debug(f"SSE event type: {event_type}")
                    continue
                elif line.startswith("data: "):
                    try:
                        data_content = line[6:].strip()  # Remove "data: " prefix
                        
                        # Skip empty data lines
                        if not data_content:
                            continue
                            
                        # Parse JSON data
                        data = json.loads(data_content)
                        logger.debug(f"SSE data: {json.dumps(data, indent=2)}")
                        
                        # Handle different response types
                        if "result" in data:
                            result = data["result"]
                            logger.info("Received result from MCP server")
                            
                            # Check for error indication in result
                            if data.get("isError", False) or (isinstance(result, dict) and result.get("isError", False)):
                                error_message = data.get("message", result.get("content", "Unknown error in final response"))
                                logger.error(f"MCP server result contains error: {error_message}")
                                
                                # Check for specific ZeroGPU quota errors
                                if isinstance(error_message, list):
                                    for error_item in error_message:
                                        if isinstance(error_item, dict) and error_item.get("type") == "text":
                                            error_text = error_item.get("text", "")
                                            if "ZeroGPU quota exceeded" in error_text:
                                                logger.error("ZeroGPU quota exceeded - consider using authentication or waiting for quota reset")
                                            elif "quota" in error_text.lower():
                                                logger.error(f"Quota-related error: {error_text}")
                                
                                return None
                            break
                        elif "error" in data:
                            # Handle standard JSON-RPC error format
                            error_info = data["error"]
                            if isinstance(error_info, dict):
                                error_msg = error_info.get("message", str(error_info))
                                error_code = error_info.get("code", "unknown")
                                logger.error(f"MCP server JSON-RPC error [{error_code}]: {error_msg}")
                            else:
                                logger.error(f"MCP server error: {error_info}")
                            return None
                        elif data.get("is_error", False) or data.get("isError", False):
                            # Handle is_error field format
                            error_message = data.get("message", data.get("error_message", "Unknown error occurred"))
                            error_details = data.get("details", data.get("error_details", ""))
                            
                            # Log the error with full details
                            if error_details:
                                logger.error(f"MCP server error: {error_message} - Details: {error_details}")
                            else:
                                logger.error(f"MCP server error: {error_message}")
                            
                            # Also log the full error response for debugging
                            logger.debug(f"Full error response: {json.dumps(data, indent=2)}")
                            return None
                        elif "id" in data and data.get("jsonrpc") == "2.0":
                            # This might be the final response
                            if "result" in data:
                                result = data["result"]
                                break
                            elif data.get("is_error", False) or data.get("isError", False):
                                # Handle is_error in final JSON-RPC response
                                error_message = data.get("message", data.get("error_message", "Unknown error in final response"))
                                logger.error(f"MCP server final response error: {error_message}")
                                return None
                        else:
                            # Might be a progress update or intermediate data
                            logger.debug(f"Intermediate SSE data: {data}")
                            
                    except json.JSONDecodeError as e:
                        logger.debug(f"Non-JSON SSE data line: {data_content}")
                        continue
                else:
                    # Handle other SSE lines (comments, etc.)
                    logger.debug(f"Other SSE line: {line}")
            
            return result
            
        except requests.exceptions.Timeout:
            logger.error(f"Timeout calling MCP server at {mcp_url}")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error calling MCP server: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error calling MCP server: {e}")
            return None
    
    def _list_mcp_tools(self, mcp_url: str, hf_token: str = None) -> Optional[List[Dict[str, Any]]]:
        """
        List available tools from an MCP server using SSE with optional authentication.
        
        Args:
            mcp_url (str): The MCP server SSE endpoint URL
            hf_token (str, optional): Hugging Face access token for authentication.
                                    If None, tries to use HF_API_TOKEN from constants.
            
        Returns:
            Optional[List[Dict[str, Any]]]: List of available tools or None if failed
        """
        try:
            # Import HF_API_TOKEN from constants if no token provided
            if not hf_token:
                from modules.constants import HF_API_TOKEN
                hf_token = HF_API_TOKEN
                if not hf_token:
                    logger.debug("No Hugging Face access token available for tools listing")
            
            mcp_request = {
                "jsonrpc": "2.0",
                "id": f"list_tools_{int(time.time())}",
                "method": "tools/list",
                "params": {}
            }
            
            # Prepare headers with optional authentication
            headers = {
                "Accept": "application/json, text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "application/json"
            }
            
            # Add authentication header if token is available
            if hf_token:
                headers["Authorization"] = f"Bearer {hf_token}"
                logger.debug("Added Hugging Face authentication to tools list request")
            
            response = requests.post(
                mcp_url,
                json=mcp_request,
                headers=headers,
                stream=True,
                timeout=30
            )
            
            if response.status_code != 200:
                logger.debug(f"Tools list request failed with status {response.status_code}")
                if response.status_code == 401 and hf_token:
                    logger.warning("Authentication failed while listing tools - check your HF_TOKEN")
                elif response.status_code == 429:
                    logger.warning("Rate limit exceeded while listing tools")
                return None
            
            # Parse SSE response for tools list
            for line in response.iter_lines(decode_unicode=True):
                if line.startswith("data: "):
                    try:
                        data_content = line[6:].strip()
                        if not data_content:
                            continue
                            
                        data = json.loads(data_content)
                        if "result" in data and "tools" in data["result"]:
                            tools = data["result"]["tools"]
                            # Cache the tool schemas for later use
                            cache_key = f"{mcp_url}_tools"
                            self._tool_schemas_cache[cache_key] = tools
                            return tools
                        elif "result" in data and isinstance(data["result"], list):
                            tools = data["result"]
                            # Cache the tool schemas for later use
                            cache_key = f"{mcp_url}_tools"
                            self._tool_schemas_cache[cache_key] = tools
                            return tools
                            
                    except json.JSONDecodeError:
                        continue
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to list MCP tools via SSE: {e}")
            return None
    
    def get_tool_schema(self, mcp_url: str, tool_name: str) -> Optional[Dict[str, Any]]:
        """
        Get the parameter schema for a specific MCP tool.
        
        Args:
            mcp_url (str): The MCP server endpoint URL
            tool_name (str): Name of the tool to get schema for
            
        Returns:
            Optional[Dict[str, Any]]: Tool schema with parameters, or None if not found
        """
        try:
            # Check cache first
            cache_key = f"{mcp_url}_tools"
            tools = self._tool_schemas_cache.get(cache_key)
            
            # If not in cache, fetch tools list
            if tools is None:
                tools = self._list_mcp_tools(mcp_url)
                if tools is None:
                    return None
            
            # Find the specific tool
            for tool in tools:
                if tool.get("name") == tool_name:
                    logger.info(f"Found schema for tool '{tool_name}': {tool.get('description', 'No description')}")
                    if "inputSchema" in tool:
                        logger.debug(f"Tool parameters: {json.dumps(tool['inputSchema'], indent=2)}")
                        return tool["inputSchema"]
                    elif "parameters" in tool:
                        logger.debug(f"Tool parameters: {json.dumps(tool['parameters'], indent=2)}")
                        return tool["parameters"]
                    else:
                        logger.debug(f"Tool schema: {json.dumps(tool, indent=2)}")
                        return tool
            
            logger.warning(f"Tool '{tool_name}' not found on MCP server {mcp_url}")
            return None
            
        except Exception as e:
            logger.error(f"Failed to get tool schema for '{tool_name}': {e}")
            return None
    
    def validate_tool_arguments(self, mcp_url: str, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and optimize tool arguments based on the tool's schema.
        
        Args:
            mcp_url (str): The MCP server endpoint URL
            tool_name (str): Name of the tool
            arguments (Dict[str, Any]): Arguments to validate
            
        Returns:
            Dict[str, Any]: Validated and potentially modified arguments
        """
        try:
            schema = self.get_tool_schema(mcp_url, tool_name)
            if schema is None:
                logger.debug(f"No schema available for tool '{tool_name}', using arguments as-is")
                # Still filter out None values to prevent validation errors
                filtered_args = {k: v for k, v in arguments.items() if v is not None}
                return filtered_args
            
            validated_args = arguments.copy()
            
            # Extract parameter definitions from schema
            properties = schema.get("properties", {})
            if not properties:
                # Try alternative schema formats
                if "type" in schema and schema["type"] == "object":
                    properties = schema.get("properties", {})
                else:
                    logger.debug(f"No properties found in schema for '{tool_name}'")
                    # Still filter out None values
                    filtered_args = {k: v for k, v in arguments.items() if v is not None}
                    return filtered_args
            
            # Log available parameters for debugging
            logger.debug(f"Available parameters for '{tool_name}': {list(properties.keys())}")
            
            # Remove arguments not defined in schema or with None values
            schema_params = set(properties.keys())
            provided_params = set(arguments.keys())
            
            # Remove invalid parameters and None values
            invalid_params = provided_params - schema_params
            for param in invalid_params:
                logger.debug(f"Removing invalid parameter '{param}' for tool '{tool_name}'")
                validated_args.pop(param, None)
            
            # Remove None values to prevent validation errors
            none_params = [k for k, v in validated_args.items() if v is None]
            for param in none_params:
                logger.debug(f"Removing None value parameter '{param}' for tool '{tool_name}'")
                validated_args.pop(param, None)
            
            # Add default values for missing required parameters
            required_params = schema.get("required", [])
            for param in required_params:
                if param not in validated_args:
                    param_schema = properties.get(param, {})
                    default_value = param_schema.get("default")
                    if default_value is not None:
                        logger.debug(f"Adding default value for required parameter '{param}': {default_value}")
                        validated_args[param] = default_value
                    else:
                        logger.warning(f"Required parameter '{param}' missing for tool '{tool_name}' and no default value available")
            
            return validated_args
            
        except Exception as e:
            logger.error(f"Failed to validate arguments for tool '{tool_name}': {e}")
            # Return filtered arguments without None values as fallback
            filtered_args = {k: v for k, v in arguments.items() if v is not None}
            return filtered_args
    
    def generate_badge_image(
        self,
        prompt: str,
        server_key: str = "flux_lora_dlc",
        image_input: Optional[str] = None,
        width: int = 512,
        height: int = 512,
        hf_token: str = None,
        **kwargs
    ) -> Optional[Union[str, Image.Image]]:
        """
        Generate a badge image using the specified MCP server.
        
        Args:
            prompt (str): Text prompt for image generation
            server_key (str): Which server to use for generation
            image_input (Optional[str]): Path to input image for image-to-image generation
            width (int): Image width (default: 512)
            height (int): Image height (default: 512)
            hf_token (str, optional): Hugging Face access token for authentication
            **kwargs: Additional server-specific parameters
            
        Returns:
            Optional[Union[str, Image.Image]]: Generated image path or PIL Image, None if failed
        """
        try:
            if server_key not in self.servers:
                logger.error(f"Unknown server key: {server_key}. Available: {list(self.servers.keys())}")
                return None
            
            server_config = self.servers[server_key]
            mcp_url = server_config["mcp_url"]
            
            # Prepare parameters based on server type
            if server_key == "flux_lora_dlc":
                return self._generate_flux_lora_dlc_mcp(mcp_url, prompt, image_input, width, height, hf_token, **kwargs)
            elif server_key == "flux_schnell":
                return self._generate_flux_schnell_mcp(mcp_url, prompt, image_input, width, height, hf_token, **kwargs)
            elif server_key == "qwen_image_diffusion":
                return self._generate_qwen_image_diffusion_mcp(mcp_url, prompt, image_input, width, height, hf_token, **kwargs)
            else:
                logger.error(f"No MCP generation method implemented for server: {server_key}")
                return None
                
        except Exception as e:
            logger.error(f"Error generating image with {server_key}: {str(e)}")
            return None
    
    def _generate_flux_lora_dlc_mcp(
        self,
        mcp_url: str,
        prompt: str,
        image_input:Optional[str] = None,
        width: int = 512,
        height: int = 512,
        hf_token: str = None,
        **kwargs
    ) -> Optional[Union[str, Image.Image]]:
        """Generate image using FLUX LoRA DLC MCP server via SSE."""
        try:
            # First, list available tools to understand the interface
            tools = self._list_mcp_tools(mcp_url, hf_token)
            if tools:
                logger.info(f"Available tools on FLUX LoRA DLC: {[tool.get('name', 'unnamed') for tool in tools]}")
            
            # Arguments for FLUX LoRA DLC via MCP
            arguments = {
                "prompt": prompt,
                "image_input": image_input,
                "image_strength": kwargs.get("image_strength", 0.75),
                "cfg_scale": kwargs.get("cfg_scale", 3.5),
                "steps": kwargs.get("steps", 28),
                "randomize_seed": kwargs.get("randomize_seed", True),
                "selected_index": "155",
                "seed": kwargs.get("seed", 662496688),
                "width": width,
                "height": height,
                "lora_scale": kwargs.get("lora_scale", 0.95)
            }
            
            logger.info(f"Generating image with FLUX LoRA DLC MCP: {prompt[:50]}...")
            
            # Try different possible tool names for FLUX LoRA DLC
            possible_tool_names = [
                "FLUX_LoRA_DLC_run_lora",
                "run_lora"
            ]
            
            for tool_name in possible_tool_names:
                logger.info(f"Trying tool name: {tool_name}")
                
                # Validate arguments against tool schema
                validated_args = self.validate_tool_arguments(mcp_url, tool_name, arguments)
                
                result = self._call_mcp_server(mcp_url, tool_name, validated_args, hf_token)
                
                if result is not None:
                    # Try to extract PIL image from byte string responses first
                    pil_image = self._extract_image_from_mcp_result(result)
                    if pil_image:
                        logger.info(f"Successfully converted MCP result to PIL image: {pil_image.size}")
                        return pil_image
                    
                    # Fall back to original file path handling
                    if isinstance(result, dict):
                        if "content" in result:
                            # Extract file path from content
                            for content_item in result["content"]:
                                if content_item.get("type") == "image":
                                    image_path = content_item.get("data")
                                    if image_path and isinstance(image_path, str) and image_path.startswith(('/tmp/', './')):
                                        logger.info(f"Successfully got image path: {image_path}")
                                        return image_path
                        elif "image_path" in result:
                            image_path = result["image_path"]
                            logger.info(f"Successfully got image path: {image_path}")
                            return image_path
                        elif "data" in result:
                            # Handle data field
                            image_path = result["data"]
                            if image_path and isinstance(image_path, str) and image_path.startswith(('/tmp/', './')):
                                logger.info(f"Successfully got image path: {image_path}")
                                return image_path
                    elif isinstance(result, (str, list)):
                        # Handle direct file path response
                        image_path = result[0] if isinstance(result, list) else result
                        if image_path and isinstance(image_path, str) and image_path.startswith(('/tmp/', './')):
                            logger.info(f"Successfully got image path: {image_path}")
                            return image_path
                    
                    logger.debug(f"Tool {tool_name} returned result but format not recognized: {type(result)}")
                    logger.debug(f"Result content: {result}")
                else:
                    logger.debug(f"Tool {tool_name} returned None")
                    
            logger.warning("No valid result returned from FLUX LoRA DLC MCP")
            return None
                
        except Exception as e:
            logger.error(f"FLUX LoRA DLC MCP generation failed: {str(e)}")
            return None
    
    def _generate_flux_schnell_mcp(
        self,
        mcp_url: str,
        prompt: str,
        image_input: Optional[str] = None,
        width: int = 512,
        height: int = 512,
        hf_token: str = None,
        **kwargs
    ) -> Optional[Union[str, Image.Image]]:
        """Generate image using FLUX Schnell MCP server via SSE."""
        try:
            # First, list available tools to understand the interface
            tools = self._list_mcp_tools(mcp_url, hf_token)
            if tools:
                logger.info(f"Available tools on FLUX Schnell: {[tool.get('name', 'unnamed') for tool in tools]}")
            
            # Arguments for FLUX Schnell via MCP
            arguments = {
                "prompt": prompt,
                "width": width,
                "height": height,
                "num_inference_steps": kwargs.get("num_inference_steps", 4),  # Schnell is fast, fewer steps
                "randomize_seed": kwargs.get("randomize_seed", True),
                "seed": kwargs.get("seed", 0)
            }
            
            # Only add image input parameters if image_input is provided and not None
            if image_input is not None:
                arguments["image_input"] = image_input
                arguments["strength"] = kwargs.get("strength", 0.75)
            
            logger.info(f"Generating image with FLUX Schnell MCP: {prompt[:50]}...")
            
            # Try possible tool names for FLUX Schnell
            possible_tool_names = ["flux1_schnell_infer", "infer"]
            
            for tool_name in possible_tool_names:
                logger.info(f"Trying tool name: {tool_name}")
                
                # Validate arguments against tool schema
                validated_args = self.validate_tool_arguments(mcp_url, tool_name, arguments)
                
                result = self._call_mcp_server(mcp_url, tool_name, validated_args, hf_token)
                
                if result is not None:
                    # Try to extract PIL image from byte string responses first
                    pil_image = self._extract_image_from_mcp_result(result)
                    if pil_image:
                        logger.info(f"Successfully converted MCP result to PIL image: {pil_image.size}")
                        return pil_image
                    
                    # Fall back to original file path handling
                    if isinstance(result, dict):
                        if "content" in result:
                            # Extract file path from content
                            for content_item in result["content"]:
                                if content_item.get("type") == "image":
                                    image_path = content_item.get("data")
                                    if image_path and isinstance(image_path, str) and image_path.startswith(('/tmp/', './')):
                                        logger.info(f"Successfully got image path: {image_path}")
                                        return image_path
                        elif "image_path" in result:
                            image_path = result["image_path"]
                            logger.info(f"Successfully got image path: {image_path}")
                            return image_path
                        elif "data" in result:
                            image_path = result["data"]
                            if image_path and isinstance(image_path, str) and image_path.startswith(('/tmp/', './')):
                                logger.info(f"Successfully got image path: {image_path}")
                                return image_path
                    elif isinstance(result, (str, list)):
                        image_path = result[0] if isinstance(result, list) else result
                        if image_path and isinstance(image_path, str) and image_path.startswith(('/tmp/', './')):
                            logger.info(f"Successfully got image path: {image_path}")
                            return image_path
                    
                    logger.debug(f"Tool {tool_name} returned result but could not extract image: {type(result)}")
            
            logger.warning("No valid result returned from FLUX Schnell MCP")
            return None
                
        except Exception as e:
            logger.error(f"FLUX Schnell MCP generation failed: {str(e)}")
            return None
    
    def _generate_qwen_image_diffusion_mcp(
        self,
        mcp_url: str,
        prompt: str,
        image_input: Optional[str] = None,
        width: int = 512,
        height: int = 512,
        hf_token: str = None,
        **kwargs
    ) -> Optional[Union[str, Image.Image]]:
        """Generate image using Qwen Image Diffusion MCP server via SSE."""
        try:
            # First, list available tools to understand the interface
            tools = self._list_mcp_tools(mcp_url, hf_token)
            if tools:
                logger.info(f"Available tools on Qwen Image Diffusion: {[tool.get('name', 'unnamed') for tool in tools]}")
            
            # Arguments for Qwen Image Diffusion via MCP with enhanced parameters
            arguments = {
                "prompt": prompt,
                "width": width,
                "height": height,
                "guidance_scale": kwargs.get("guidance_scale", 4.0),
                "num_inference_steps": kwargs.get("num_inference_steps", 40),
                "seed": kwargs.get("seed", 0),
                "randomize_seed": kwargs.get("randomize_seed", True),
                # Enhanced parameters for Qwen Image Diffusion
                "num_images": kwargs.get("num_images", 1),
                "zip_images": kwargs.get("zip_images", False),
                "use_negative_prompt": kwargs.get("use_negative_prompt", True),
                "negative_prompt": kwargs.get("negative_prompt", badge_negative_prompt)  # 🎯 Uses constants
            }
            
            # Only add image input parameters if image_input is provided and not None
            if image_input is not None:
                arguments["image_input"] = image_input
                arguments["strength"] = kwargs.get("strength", 0.75)
            
            logger.info(f"Generating image with Qwen Image Diffusion MCP: {prompt[:50]}")
            logger.debug(f"Using negative prompt: {arguments['negative_prompt']}")
            
            # Try possible tool names for Qwen Image Diffusion
            possible_tool_names = ["Qwen_Image_Diffusion_generate", "generate"]
            
            for tool_name in possible_tool_names:
                logger.info(f"Trying tool name: {tool_name}")
                
                # Validate arguments against tool schema
                validated_args = self.validate_tool_arguments(mcp_url, tool_name, arguments)
                
                result = self._call_mcp_server(mcp_url, tool_name, validated_args, hf_token)
                
                if result is not None:
                    # Try to extract PIL image from byte string responses first
                    pil_image = self._extract_image_from_mcp_result(result)
                    if pil_image and isinstance(pil_image, Image.Image):
                        logger.info(f"Successfully converted MCP result to PIL image: {pil_image.size}")
                        return pil_image
                    
                    # Fall back to original file path handling
                    if isinstance(result, dict):
                        if "content" in result:
                            # Extract file path from content
                            for content_item in result["content"]:
                                if content_item.get("type") == "text":
                                    content_text = content_item.get("text")
                                    image_path = content_text["url"] if isinstance(content_text, dict) else content_text
                                    return image_path
                                if content_item.get("type") == "image":
                                    image_path = content_item.get("data")
                                    if image_path and isinstance(image_path, str) and image_path.startswith(('/tmp/', './')):
                                        logger.info(f"Successfully got image path: {image_path}")
                                        return image_path
                        elif "image_path" in result:
                            image_path = result["image_path"]
                            logger.info(f"Successfully got image path: {image_path}")
                            return image_path
                        elif "data" in result:
                            image_path = result["data"]
                            if image_path and isinstance(image_path, str) and image_path.startswith(('/tmp/', './')):
                                logger.info(f"Successfully got image path: {image_path}")
                                return image_path
                    elif isinstance(result, (str, list)):
                        image_path = result[0] if isinstance(result, list) else result
                        if image_path and isinstance(image_path, str) and image_path.startswith(('/tmp/', './')):
                            logger.info(f"Successfully got image path: {image_path}")
                            return image_path
                    
                    logger.debug(f"Tool {tool_name} returned result but could not extract image: {type(result)}")
            
            logger.warning("No valid result returned from Qwen Image Diffusion MCP")
            return None
                
        except Exception as e:
            logger.error(f"Qwen Image Diffusion MCP generation failed: {str(e)}")
            return None
    
    def generate_badge_with_fallback(
        self,
        prompt: str,
        preferred_servers: list = None,
        image_input: Optional[str] = None,
        width: int = 512,
        height: int = 512,
        hf_token: str = None,
        **kwargs
    ) -> Optional[Union[str, Image.Image]]:
        """
        Generate badge image with automatic fallback between MCP servers using SSE.
        Properly handles both file path and byte string responses from MCP servers.
        
        Args:
            prompt (str): Text prompt for image generation
            preferred_servers (list): List of server keys to try in order
            image_input (Optional[str]): Path to input image for image-to-image
            width (int): Image width
            height (int): Image height
            hf_token (str, optional): Hugging Face access token for authentication
            **kwargs: Additional parameters
            
        Returns:
            Optional[Union[str, Image.Image]]: Generated image (file path string or PIL Image) or None if all MCP servers failed
        """
        if preferred_servers is None:
            preferred_servers = ["flux_lora_dlc", "flux_schnell", "qwen_image_diffusion"]
        
        logger.info(f"Attempting badge generation with MCP fallback servers: {preferred_servers}")
        
        for server_key in preferred_servers:
            logger.info(f"Trying MCP server: {server_key}")
            result = self.generate_badge_image(
                prompt=prompt,
                server_key=server_key,
                image_input=image_input,
                width=width,
                height=height,
                hf_token=hf_token,
                **kwargs
            )
            
            if result is not None:
                if isinstance(result, str):
                    logger.info(f"Successfully generated image file path using MCP server {server_key}: {result}")
                elif hasattr(result, 'size'):
                    logger.info(f"Successfully generated PIL image using MCP server {server_key}: {result.size}")
                else:
                    logger.info(f"Successfully generated image using MCP server {server_key}: {type(result)}")
                return result
            else:
                logger.warning(f"MCP server {server_key} failed, trying next...")
        
        logger.error("All MCP servers failed to generate image")
        return None
    
    def create_badge_prompt(
        self,
        achievement_name: str,
        recipient_name: str = "",
        style: str = "professional",
        colors: str = "gold, blue, white",
        elements: str = "stars, ribbons, laurel wreaths"
    ) -> str:
        """
        Create an optimized prompt for badge generation.
        
        Args:
            achievement_name (str): Name of the achievement
            recipient_name (str): Name of the recipient (optional)
            style (str): Style of the badge (professional, modern, artistic, classic, superhero, retro)
            colors (str): Color scheme
            elements (str): Decorative elements to include
            
        Returns:
            str: Optimized prompt for badge generation
        """       
        if style in STYLE_TEMPLATES:
            template_func = STYLE_TEMPLATES[style]
            return template_func(achievement_name, recipient_name, colors, elements)
        else:
            # Fallback to professional style if unknown style is requested
            template_func = STYLE_TEMPLATES["professional"]
            return template_func(achievement_name, recipient_name, colors, elements)
    
    def test_mcp_connection(self, server_key: str, hf_token: str = None) -> bool:
        """
        Test connection to an MCP server using SSE with optional authentication.
        
        Args:
            server_key (str): Server key to test
            hf_token (str, optional): Hugging Face access token for authentication
            
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            if server_key not in self.servers:
                logger.error(f"Unknown server key: {server_key}")
                return False
            
            server_config = self.servers[server_key]
            mcp_url = server_config["mcp_url"]
            
            # Test with tools/list request via SSE
            tools = self._list_mcp_tools(mcp_url, hf_token)
            
            if tools is not None:
                logger.info(f"MCP server {server_key} is responding with {len(tools)} tools")
                for tool in tools:
                    tool_name = tool.get('name', 'unnamed')
                    tool_desc = tool.get('description', 'no description')
                    logger.info(f"  - {tool_name}: {tool_desc}")
                    
                    # Also log available parameters for debugging
                    schema = self.get_tool_schema(mcp_url, tool_name)
                    if schema and "properties" in schema:
                        params = list(schema["properties"].keys())
                        logger.debug(f"    Parameters: {params}")
                return True
            else:
                logger.warning(f"MCP server {server_key} did not return tools list")
                return False
                
        except Exception as e:
            logger.error(f"Failed to test MCP server {server_key}: {e}")
            return False
    
    def get_available_servers(self) -> Dict[str, str]:
        """
        Get list of available MCP servers and their descriptions.
        
        Returns:
            Dict[str, str]: Dictionary of server keys and descriptions
        """
        return {key: config["description"] for key, config in self.servers.items()}
    
    def _convert_bytes_to_pil_image(self, data: Union[str, bytes], format_hint: str = "webp") -> Optional[Image.Image]:
        """
        Convert byte string data to PIL Image in PNG format.
        
        Args:
            data (Union[str, bytes]): Image data as byte string or base64 encoded string
            format_hint (str): Hint about the original format (e.g., "webp", "png", "jpg")
            
        Returns:
            Optional[Image.Image]: PIL Image object or None if conversion failed
        """
        try:
            image_bytes = None
            
            # Handle different data types
            if isinstance(data, str):
                # Try to decode as base64
                try:
                    # Remove data URL prefix if present (data:image/webp;base64,...)
                    if data.startswith('data:'):
                        # Find the base64 part after the comma
                        comma_pos = data.find(',')
                        if comma_pos != -1:
                            data = data[comma_pos + 1:]
                    
                    image_bytes = base64.b64decode(data)
                    logger.debug(f"Successfully decoded base64 data, size: {len(image_bytes)} bytes")
                except Exception as e:
                    logger.debug(f"Failed to decode as base64: {e}")
                    # Try to treat as raw byte string
                    try:
                        image_bytes = data.encode('latin1') if isinstance(data, str) else data
                        logger.debug(f"Treating as raw byte string, size: {len(image_bytes)} bytes")
                    except Exception as e2:
                        logger.error(f"Failed to convert string to bytes: {e2}")
                        return None
                        
            elif isinstance(data, bytes):
                image_bytes = data
                logger.debug(f"Using raw bytes data, size: {len(image_bytes)} bytes")
            else:
                logger.error(f"Unsupported data type: {type(data)}")
                return None
            
            if image_bytes is None:
                logger.error("Failed to extract image bytes from data")
                return None
            
            # Create BytesIO object and open with PIL
            image_buffer = io.BytesIO(image_bytes)
            
            # Try to open the image
            try:
                pil_image = Image.open(image_buffer)
                logger.info(f"Successfully opened image: {pil_image.format} {pil_image.size} {pil_image.mode}")
                
                # Convert to RGBA for transparency support and ensure PNG compatibility
                if pil_image.mode not in ['RGB', 'RGBA']:
                    if pil_image.mode == 'P' and 'transparency' in pil_image.info:
                        # Handle palette mode with transparency
                        pil_image = pil_image.convert('RGBA')
                    elif pil_image.mode in ['L', 'LA']:
                        # Handle grayscale modes
                        pil_image = pil_image.convert('RGBA')
                    else:
                        # Convert other modes to RGBA
                        pil_image = pil_image.convert('RGBA')
                        
                logger.info(f"Converted image to mode: {pil_image.mode}")
                return pil_image
                
            except Exception as e:
                logger.error(f"Failed to open image with PIL: {e}")
                
                # Log first few bytes for debugging
                if len(image_bytes) > 20:
                    header_hex = ' '.join(f'{b:02x}' for b in image_bytes[:20])
                    logger.debug(f"Image data header (first 20 bytes): {header_hex}")
                
                return None
                
        except Exception as e:
            logger.error(f"Error converting bytes to PIL image: {e}")
            return None
    
    def _extract_image_from_mcp_result(self, result: Any) -> Optional[Image.Image]:
        """
        Extract PIL Image from MCP server result, handling various response formats.
        
        Args:
            result: MCP server response result
            
        Returns:
            Optional[Image.Image]: PIL Image object or None if extraction failed
        """
        try:
            if isinstance(result, dict):
                if "content" in result:
                    # Handle content array format
                    for content_item in result["content"]:
                        if isinstance(content_item, dict):
                            if content_item.get("type") == "image":
                                # Handle byte string image data
                                image_data = content_item.get("data")
                                if image_data:
                                    # Determine format from mimeType if available
                                    mime_type = content_item.get("mimeType", "image/webp")
                                    format_hint = "webp" if "webp" in mime_type else "png" if "png" in mime_type else "jpg" if "jpeg" in mime_type or "jpg" in mime_type else "webp"
                                    
                                    logger.info(f"Found image data in content with mimeType: {mime_type}")
                                    pil_image = self._convert_bytes_to_pil_image(image_data, format_hint)
                                    if pil_image:
                                        logger.info(f"Successfully converted MCP result to PIL image: {pil_image.size}")
                                        return pil_image
                                        
                            elif content_item.get("type") == "text":
                                # Handle text responses that may contain image URLs or structured data
                                text_content = content_item.get("text", "")
                                
                                # Handle structured data in string format like "[{'image': {'url': '...'}}, ...]"
                                if text_content.startswith("[") and "url" in text_content:
                                    try:
                                        # Parse the string as a Python literal (safely evaluate)
                                        import ast
                                        parsed_data = ast.literal_eval(text_content)
                                        
                                        # Extract URL from the nested structure
                                        if isinstance(parsed_data, list) and len(parsed_data) > 0:
                                            first_item = parsed_data[0]
                                            if isinstance(first_item, dict) and 'image' in first_item:
                                                image_info = first_item['image']
                                                if isinstance(image_info, dict) and 'url' in image_info:
                                                    image_path = image_info['url']
                                                    logger.info(f"Successfully extracted URL from structured text: {image_path}")

                                                    # Try to download and convert the image from URL
                                                    try:
                                                        import requests
                                                        response = requests.get(image_path, timeout=30)
                                                        response.raise_for_status()
                                            
                                                        # Convert downloaded bytes to PIL image
                                                        pil_image = self._convert_bytes_to_pil_image(response.content)
                                                        if pil_image:
                                                            logger.info(f"Successfully downloaded and converted image from URL: {pil_image.size}")
                                                            return pil_image
                                                
                                                    except Exception as url_error:
                                                        logger.warning(f"Failed to download image from URL {image_path}: {url_error}")
                                                    return image_path
                                    
                                    except (ValueError, SyntaxError, KeyError) as e:
                                        logger.debug(f"Failed to parse structured text content as literal: {e}")
                                        # Fall through to other text processing methods
                                
                                # Handle direct URL text content like "Image URL: https://..."
                                if "Image URL:" in text_content:
                                    # Extract URL from text like "Image URL: https://..."
                                    try:
                                        url_start = text_content.find("Image URL:") + len("Image URL:").strip()
                                        image_url = text_content[url_start:].strip()
                                        
                                        # Clean up any potential extra text after the URL
                                        # URLs typically end with file extensions
                                        if any(ext in image_url.lower() for ext in ['.png', '.jpg', '.jpeg', '.webp', '.gif']):
                                            # Find the end of the URL (first whitespace or end of string)
                                            for i, char in enumerate(image_url):
                                                if char.isspace():
                                                    image_url = image_url[:i]
                                                    break
                                        
                                        logger.info(f"Found image URL in text content: {image_url}")
                                        
                                        # Try to download and convert the image from URL
                                        try:
                                            import requests
                                            response = requests.get(image_url, timeout=30)
                                            response.raise_for_status()
                                            
                                            # Convert downloaded bytes to PIL image
                                            pil_image = self._convert_bytes_to_pil_image(response.content)
                                            if pil_image:
                                                logger.info(f"Successfully downloaded and converted image from URL: {pil_image.size}")
                                                return pil_image
                                                
                                        except Exception as url_error:
                                            logger.warning(f"Failed to download image from URL {image_url}: {url_error}")
                                            
                                    except Exception as parse_error:
                                        logger.debug(f"Failed to parse image URL from text: {parse_error}")
                                
                                # Handle cases where text_content might be a dictionary-like string
                                elif isinstance(text_content, dict) and "url" in text_content:
                                    image_path = text_content["url"]
                                    logger.info(f"Successfully got image URL from dict text content: {image_path}")
                                    return image_path
                
                # Handle direct data field
                elif "data" in result:
                    image_data = result["data"]
                    if image_data:
                        logger.info("Found image data in direct data field")
                        pil_image = self._convert_bytes_to_pil_image(image_data)
                        if pil_image:
                            logger.info(f"Successfully converted MCP result to PIL image: {pil_image.size}")
                            return pil_image
                
                # Handle image_path field (existing logic)
                elif "image_path" in result:
                    image_path = result["image_path"]
                    if image_path and isinstance(image_path, str):
                        try:
                            pil_image = Image.open(image_path)
                            logger.info(f"Successfully loaded image from path: {image_path}")
                            return pil_image
                        except Exception as e:
                            logger.warning(f"Failed to load image from path {image_path}: {e}")
            
            elif isinstance(result, (str, list)):
                # Handle direct file path response (existing logic)
                image_path = result[0] if isinstance(result, list) else result
                if image_path and isinstance(image_path, str):
                    # Check if it looks like a file path vs byte data
                    if not image_path.startswith(('/tmp/', './')) and len(image_path) > 100:
                        # Likely base64 or byte data disguised as string
                        logger.info("String result appears to be image data rather than file path")
                        pil_image = self._convert_bytes_to_pil_image(image_path)
                        if pil_image:
                            logger.info(f"Successfully converted string result to PIL image: {pil_image.size}")
                            return pil_image
                    else:
                        # Treat as file path
                        try:
                            pil_image = Image.open(image_path)
                            logger.info(f"Successfully loaded image from path: {image_path}")
                            return pil_image
                        except Exception as e:
                            logger.warning(f"Failed to load image from path {image_path}: {e}")
            
            logger.warning("Could not extract image from MCP result")
            return None
            
        except Exception as e:
            logger.error(f"Error extracting image from MCP result: {e}")
            return None


# Global instance for easy access
mcp_generator = MCPImageGenerator()

# Convenience functions for backward compatibility and easy use
def generate_badge_image(
    prompt: str,
    server_key: str = "flux_lora_dlc",
    image_input: Optional[str] = None,
    width: int = 512,
    height: int = 512,
    hf_token: str = None,
    **kwargs
) -> Optional[Union[str, Image.Image]]:
    """
    Convenience function to generate badge image using the global MCP generator with SSE.
    
    Args:
        prompt (str): Text prompt for image generation
        server_key (str): Which MCP server to use for generation
        image_input (Optional[str]): Path to input image for image-to-image
        width (int): Image width (default: 512)
        height (int): Image height (default: 512)
        hf_token (str, optional): Hugging Face access token for authentication
        **kwargs: Additional server-specific parameters
        
    Returns:
        Optional[Union[str, Image.Image]]: Generated image path or PIL Image, None if failed
    """
    return mcp_generator.generate_badge_image(
        prompt=prompt,
        server_key=server_key,
        image_input=image_input,
        width=width,
        height=height,
        hf_token=hf_token,
        **kwargs
    )

def create_badge_prompt(
    achievement_name: str,
    recipient_name: str = "",
    style: str = "professional",
    colors: str = "gold, blue, white",
    elements: str = "stars, ribbons, laurel wreaths"
    ) -> str:
    """
    Convenience function to create an optimized prompt for badge generation.
    Args:
        achievement_name (str): Name of the achievement
        recipient_name (str): Name of the recipient (optional)
        style (str): Style of the badge (professional, modern, artistic, classic, superhero, retro)
        colors (str): Color scheme
        elements (str): Decorative elements to include
    Returns:
        str: Optimized prompt for badge generation
    """
    return mcp_generator.create_badge_prompt(
        achievement_name=achievement_name,
        recipient_name=recipient_name,
        style=style,
        colors=colors,
        elements=elements
    )


def generate_badge_with_fallback(
    prompt: str,
    preferred_servers: list = None,
    hf_token: str = None,
    **kwargs
) -> Optional[Union[str, Image.Image]]:
    """
    Convenience function to generate badge with automatic MCP server fallback.
    Properly handles both file path and byte string responses from MCP servers.
    
    Args:
        prompt (str): Text prompt for image generation
        preferred_servers (list): List of MCP server keys to try in order
        hf_token (str, optional): Hugging Face access token for authentication
        **kwargs: Additional parameters
        
    Returns:
        Optional[Union[str, Image.Image]]: Generated image (file path string or PIL Image) or None if all MCP servers failed
    """
    return mcp_generator.generate_badge_with_fallback(
        prompt=prompt,
        preferred_servers=preferred_servers,
        hf_token=hf_token,
        **kwargs
    )

def test_mcp_servers(hf_token: str = None) -> Dict[str, bool]:
    """
    Test connections to all available MCP servers.
    
    Args:
        hf_token (str, optional): Hugging Face access token for authentication
        
    Returns:
        Dict[str, bool]: Dictionary mapping server keys to connection status
    """
    connection_results = {}
    
    for server_key in mcp_generator.servers.keys():
        try:
            is_connected = mcp_generator.test_mcp_connection(server_key, hf_token)
            connection_results[server_key] = is_connected
        except Exception as e:
            logger.error(f"Error testing {server_key}: {e}")
            connection_results[server_key] = False
    
    return connection_results

# Example usage and testing
if __name__ == "__main__":
    # Example usage
    generator = MCPImageGenerator()
    
    # Test MCP server connections with authentication
    print("Testing MCP server connections:")
    connection_results = test_mcp_servers()
    for server, status in connection_results.items():
        print(f"  {server}: {'✅ Connected' if status else '❌ Failed'}")
    
    # List available MCP servers
    print("\nAvailable MCP servers:")
    for key, desc in generator.get_available_servers().items():
        print(f"  {key}: {desc}")
    
    # Test tool schema retrieval
    print("\nTesting tool schema retrieval:")
    for server_key in ["flux_lora_dlc", "qwen_image_diffusion"]:
        print(f"\n{server_key.upper()} Server Tools:")
        tools = generator._list_mcp_tools(generator.servers[server_key]["mcp_url"])
        if tools:
            for tool in tools[:2]:  # Show first 2 tools
                tool_name = tool.get("name", "unnamed")
                schema = generator.get_tool_schema(generator.servers[server_key]["mcp_url"], tool_name)
                if schema and "properties" in schema:
                    params = list(schema["properties"].keys())[:5]  # Show first 5 params
                    print(f"  {tool_name}: {params}...")

    # Create a shared test prompt
    prompt = generator.create_badge_prompt(
        achievement_name="Open Badge MCP Certification",
        recipient_name="Surn",
        style="professional"
    )
    print(f"\nGenerated prompt: {prompt}")
    
    # Test generate_badge_with_fallback separately for each of the three preferred servers
    print("\n🔬 Testing generate_badge_with_fallback individually for each server:")
    
    preferred_servers = ["flux_lora_dlc", "flux_schnell", "qwen_image_diffusion"]
    
    for server_key in preferred_servers:
        print(f"\n--- Testing {server_key.upper()} ---")
        try:
            result = generate_badge_with_fallback(
                prompt=prompt,
                preferred_servers=[server_key],  # Test only this single server
                hf_token=HF_API_TOKEN,
                use_negative_prompt=True,
                negative_prompt=badge_negative_prompt,
                num_images=1
            )
            
            if result:
                if isinstance(result, str):
                    print(f"✅ {server_key} generated badge successfully (file path): {result}")
                elif hasattr(result, 'size'):
                    print(f"✅ {server_key} generated badge successfully (PIL image): {result.size} {result.mode}")
                    result.show()
                else:
                    print(f"✅ {server_key} generated badge successfully: {type(result)}")
            else:
                print(f"❌ {server_key} failed to generate badge")
                
        except Exception as e:
            print(f"❌ Error testing {server_key}: {e}")
    
    # Test with all servers in fallback mode
    # print(f"\n--- Testing Full Fallback Chain ---")
    # try:
    #     result = generate_badge_with_fallback(
    #         prompt=prompt,
    #         preferred_servers=preferred_servers,
    #         hf_token=HF_API_TOKEN,
    #         use_negative_prompt=True,
    #         negative_prompt=badge_negative_prompt,
    #         num_images=1
    #     )
        
    #     if result:
    #         if isinstance(result, str):
    #             print(f"✅ Full fallback chain generated badge successfully (file path): {result}")
    #         elif hasattr(result, 'size'):
    #             print(f"✅ Full fallback chain generated badge successfully (PIL image): {result.size} {result.mode}")
    #         else:
    #             print(f"✅ Full fallback chain generated badge successfully: {type(result)}")
    #     else:
    #         print("❌ All servers in fallback chain failed to generate badge")
            
    # except Exception as e:
    #     print(f"❌ Error testing full fallback chain: {e}")
    
    # Authentication status
    print(f"\n🔑 Authentication Status:")
    try:
        if HF_API_TOKEN:
            print("✅ HF_TOKEN is available for authenticated requests")
        else:
            print("❌ HF_TOKEN not found - may encounter quota limitations")
    except Exception as e:
        print(f"❌ Error checking HF_TOKEN: {e}")

