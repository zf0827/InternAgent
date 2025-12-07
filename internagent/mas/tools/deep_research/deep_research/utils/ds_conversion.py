import json
import re
import uuid
import inspect
from datetime import datetime


def convert_dsml_tool_calls_to_openai_format(completion):
    """
    Convert DeepSeek-V3.2 style DSML tool calls to OpenAI format
    
    How it works:
    1. Read the content field from completion
    2. Remove all "｜DSML｜" (full-width characters)
    3. Parse XML format tool call blocks
    4. Convert to OpenAI format tool_calls
    5. Remove tool call blocks from content
    6. Return the processed completion
    
    Args:
        completion: OpenAI ChatCompletion object
        
    Returns:
        Processed completion object with tool_calls field
    """
    # Check if choices exist
    if not hasattr(completion, 'choices') or not completion.choices:
        return completion
    
    choice = completion.choices[0]
    if not hasattr(choice, 'message'):
        return completion
    
    message = choice.message
    
    # If tool_calls already exist, no conversion needed
    if hasattr(message, 'tool_calls') and message.tool_calls:
        return completion
    
    # Get content
    content = getattr(message, 'content', '') or ''
    if not content or not isinstance(content, str):
        return completion
    
    # Step 1: Remove all "｜DSML｜" (full-width characters)
    # This handles cases like <｜DSML｜tag> and </｜DSML｜tag>
    cleaned_content = content.replace('｜DSML｜', '')
    
    # Check if tool call blocks are present
    function_calls_pattern = r'<function_calls>(.*?)</function_calls>'
    match = re.search(function_calls_pattern, cleaned_content, re.DOTALL)
    
    if not match:
        return completion
    
    xml_content = match.group(1)
    tool_calls_list = []
    
    try:
        # Parse XML tool calls
        # Find all invoke tags
        invoke_pattern = r'<invoke\s+name=["\']([^"\']+)["\'][^>]*>(.*?)</invoke>'
        invoke_matches = re.finditer(invoke_pattern, xml_content, re.DOTALL)
        
        for invoke_match in invoke_matches:
            function_name = invoke_match.group(1)
            invoke_content = invoke_match.group(2)
            
            # Parse parameters
            parameters = {}
            parameter_pattern = r'<parameter\s+name=["\']([^"\']+)["\']\s+string=["\']([^"\']*)["\'][^>]*>(.*?)</parameter>'
            param_matches = re.finditer(parameter_pattern, invoke_content, re.DOTALL)
            
            for param_match in param_matches:
                param_name = param_match.group(1)
                param_string_flag = param_match.group(2).lower() == 'true'
                param_value = param_match.group(3).strip()
                
                # If string="true", keep as string; otherwise try to parse as JSON
                if param_string_flag:
                    parameters[param_name] = param_value
                else:
                    try:
                        # Try to parse as JSON
                        parameters[param_name] = json.loads(param_value)
                    except json.JSONDecodeError:
                        # If parsing fails, keep original string
                        parameters[param_name] = param_value
            
            if parameters:
                tool_calls_list.append({
                    'name': function_name,
                    'arguments': json.dumps(parameters, ensure_ascii=False)
                })
    
    except Exception as e:
        # Silently return original completion on parsing error
        # Error handling is done in patch_agent_client if enable_debug is True
        return completion
    
    if not tool_calls_list:
        return completion
    
    # Construct OpenAI format tool_calls
    tool_calls = []
    for idx, tool_call in enumerate(tool_calls_list):
        tool_call_id = f"call_{uuid.uuid4().hex[:20]}"
        tool_calls.append({
            'id': tool_call_id,
            'type': 'function',
            'function': {
                'name': tool_call['name'],
                'arguments': tool_call['arguments']
            }
        })
    
    # Remove tool call blocks from content
    new_content = re.sub(function_calls_pattern, '', cleaned_content, flags=re.DOTALL).strip()
    
    # Update completion object
    # Update message.content
    if hasattr(message, 'content'):
        message.content = new_content
    elif hasattr(message, '__dict__'):
        message.__dict__['content'] = new_content
    
    # Set tool_calls
    if hasattr(message, 'tool_calls'):
        message.tool_calls = tool_calls
    elif hasattr(message, '__dict__'):
        message.__dict__['tool_calls'] = tool_calls
    else:
        # Create __dict__ if it doesn't exist
        if not hasattr(message, '__dict__'):
            object.__setattr__(message, '__dict__', {})
        message.__dict__['tool_calls'] = tool_calls
    
    # Update finish_reason
    if hasattr(choice, 'finish_reason'):
        choice.finish_reason = 'tool_calls'
    elif hasattr(choice, '__dict__'):
        choice.__dict__['finish_reason'] = 'tool_calls'
    
    return completion


def _print_original_response(response, agent_name):
    """Print original response (before conversion)"""
    print("\n" + "="*80)
    print(f"[XML Conversion] Original Response (Before Conversion) - {agent_name}")
    print("="*80)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Try to convert to dict format
        if hasattr(response, 'to_dict'):
            response_dict = response.to_dict()
        elif hasattr(response, 'model_dump'):
            response_dict = response.model_dump()
        elif hasattr(response, '__dict__'):
            response_dict = response.__dict__.copy()
        else:
            response_dict = {"raw_response": str(response)}
        
        # Extract key information
        choices = response_dict.get('choices', [])
        if choices:
            message = choices[0].get('message', {})
            tool_calls = message.get('tool_calls')
            content = message.get('content', '')
            finish_reason = choices[0].get('finish_reason', '')
            
            print(f"\n📥 Response Summary:")
            print("-" * 80)
            print(f"Finish Reason: {finish_reason}")
            if content:
                content_str = str(content)
                if len(content_str) > 1000:
                    print(f"Content (first 1000 chars): {content_str[:1000]}...")
                    print(f"Content (total length): {len(content_str)} chars")
                else:
                    print(f"Content: {content_str}")
            else:
                print("Content: (empty)")
            
            if tool_calls:
                print(f"\n🔧 Tool Calls ({len(tool_calls)}):")
                print("-" * 80)
                print(json.dumps(tool_calls, indent=2, ensure_ascii=False))
            else:
                print(f"\n🔧 Tool Calls: (none)")
            
            # Check if XML format tool calls are present
            if content and isinstance(content, str):
                if '<function_calls>' in content or '｜DSML｜' in content:
                    print(f"\n⚠️  Detected XML format tool calls (conversion needed)")
                    # Show XML part
                    xml_match = re.search(r'<function_calls>.*?</function_calls>', content, re.DOTALL)
                    if xml_match:
                        xml_part = xml_match.group(0)
                        if len(xml_part) > 500:
                            print(f"XML Part (first 500 chars): {xml_part[:500]}...")
                        else:
                            print(f"XML Part: {xml_part}")
        
        # Print full response (truncated version)
        print(f"\n📋 Full Response (JSON):")
        print("-" * 80)
        response_str = json.dumps(response_dict, indent=2, ensure_ascii=False)
        if len(response_str) > 3000:
            print(response_str[:3000] + f"... (truncated, {len(response_str)} chars total)")
        else:
            print(response_str)
            
    except Exception as e:
        print(f"\n⚠️  Failed to format original response: {e}")
        print(f"Response type: {type(response).__name__}")
        print(f"Original response: {str(response)[:500]}")
    
    print("="*80)


def _print_converted_response(response, agent_name):
    """Print converted response"""
    print("\n" + "="*80)
    print(f"[XML Conversion] Converted Response - {agent_name}")
    print("="*80)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Try to convert to dict format
        if hasattr(response, 'to_dict'):
            response_dict = response.to_dict()
        elif hasattr(response, 'model_dump'):
            response_dict = response.model_dump()
        elif hasattr(response, '__dict__'):
            response_dict = response.__dict__.copy()
        else:
            response_dict = {"raw_response": str(response)}
        
        # Extract key information
        choices = response_dict.get('choices', [])
        if choices:
            message = choices[0].get('message', {})
            tool_calls = message.get('tool_calls')
            content = message.get('content', '')
            finish_reason = choices[0].get('finish_reason', '')
            
            print(f"\n📥 Response Summary:")
            print("-" * 80)
            print(f"Finish Reason: {finish_reason}")
            if content:
                content_str = str(content)
                if len(content_str) > 1000:
                    print(f"Content (first 1000 chars): {content_str[:1000]}...")
                    print(f"Content (total length): {len(content_str)} chars")
                else:
                    print(f"Content: {content_str}")
            else:
                print("Content: (empty)")
            
            # Highlight converted tool_calls
            if tool_calls:
                print(f"\n✅ Tool Calls ({len(tool_calls)}) - Converted to Standard Format:")
                print("-" * 80)
                print(json.dumps(tool_calls, indent=2, ensure_ascii=False))
            else:
                print(f"\n🔧 Tool Calls: (none)")
            
            # Check if XML still exists in content (should have been removed)
            if content and isinstance(content, str):
                if '<function_calls>' in content or '｜DSML｜' in content:
                    print(f"\n⚠️  Warning: Content still contains XML format tool calls (conversion may be incomplete)")
                else:
                    print(f"\n✓ XML tool calls in content successfully removed")
        
        # Print full response (truncated version)
        print(f"\n📋 Full Response (JSON):")
        print("-" * 80)
        response_str = json.dumps(response_dict, indent=2, ensure_ascii=False)
        if len(response_str) > 3000:
            print(response_str[:3000] + f"... (truncated, {len(response_str)} chars total)")
        else:
            print(response_str)
            
    except Exception as e:
        print(f"\n⚠️  Failed to format converted response: {e}")
        print(f"Response type: {type(response).__name__}")
        print(f"Original response: {str(response)[:500]}")
    
    print("="*80 + "\n")


def patch_agent_client(agent_instance, agent_name, convert_func, enable_debug=False):
    """
    Add XML tool call conversion for a single agent's client
    
    Args:
        agent_instance: The agent instance to patch
        agent_name: Name of the agent (for logging)
        convert_func: Function to convert DSML tool calls to OpenAI format
        enable_debug: Whether to enable debug printing
        
    Returns:
        Original create function if patched successfully, None otherwise
    """
    if not hasattr(agent_instance, 'client'):
        return None
    
    client = agent_instance.client
    if not hasattr(client, 'create'):
        return None
    
    original_create = client.create
    is_async = inspect.iscoroutinefunction(original_create)
    
    if is_async:
        # Async version
        async def create_with_xml_conversion_async(*args, **kwargs):
            """Async wrapper for create method to convert XML tool calls"""
            # Call original method to get response
            response = await original_create(*args, **kwargs)
            
            # Print original response if debug enabled
            if enable_debug:
                _print_original_response(response, agent_name)
            
            # Convert XML format tool calls
            try:
                converted_response = convert_func(response)
                
                # Print converted response if debug enabled
                if enable_debug:
                    _print_converted_response(converted_response, agent_name)
                
                return converted_response
            except Exception as e:
                if enable_debug:
                    print(f"\n[XML Conversion Warning] {agent_name} conversion failed: {e}")
                    import traceback
                    traceback.print_exc()
                # If conversion fails, return original response
                return response
        
        client.create = create_with_xml_conversion_async
    else:
        # Sync version
        def create_with_xml_conversion(*args, **kwargs):
            """Wrapper for create method to convert XML tool calls"""
            # Call original method to get response
            response = original_create(*args, **kwargs)
            
            # Print original response if debug enabled
            if enable_debug:
                _print_original_response(response, agent_name)
            
            # Convert XML format tool calls
            try:
                converted_response = convert_func(response)
                
                # Print converted response if debug enabled
                if enable_debug:
                    _print_converted_response(converted_response, agent_name)
                
                return converted_response
            except Exception as e:
                if enable_debug:
                    print(f"\n[XML Conversion Warning] {agent_name} conversion failed: {e}")
                    import traceback
                    traceback.print_exc()
                # If conversion fails, return original response
                return response
        
        client.create = create_with_xml_conversion
    
    return original_create

