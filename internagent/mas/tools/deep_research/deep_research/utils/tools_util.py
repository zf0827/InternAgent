def get_autogen_message_history(messages: list[dict[str, dict]]):
    all_messages = []
    for message in messages:
        tool_responses = message.get("tool_responses", [])
        if tool_responses:
            all_messages += tool_responses
            if message.get("role") != "tool":
                all_messages.append({key: message[key] for key in message if key != "tool_responses"})
        else:
            all_messages.append(message)
    return all_messages

