"""
LLM Client for DeepSeek API
Handles chat completions with JSON schema validation
"""

import os
import json
import time
from typing import Dict, Any, Optional, List
from openai import OpenAI
from pydantic import BaseModel, ValidationError


class LLMClient:
    """Client for DeepSeek LLM API"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize LLM client
        
        Args:
            api_key: DeepSeek API key (defaults to environment variable)
        """
        self.api_key = api_key or os.environ.get('DEEPSEEK_API_KEY')
        if not self.api_key:
            raise ValueError("Missing DEEPSEEK_API_KEY. Please set env var or pass api_key explicitly.")
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.deepseek.com",
            timeout=300.0  # 5 minutes timeout for API calls
        )
        self.model = "deepseek-chat"
        self.max_retries = 3
        self.retry_delay = 1.0
    
    def chat_completion(
        self,
        system_prompt: str,
        user_message: str,
        output_schema: Optional[BaseModel] = None,
        temperature: float = 0.3,
        max_tokens: int = 4000
    ) -> Dict[str, Any]:
        """
        Send chat completion request
        
        Args:
            system_prompt: System prompt
            user_message: User message
            output_schema: Optional Pydantic model for output validation
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        
        Returns:
            Parsed JSON response (validated if schema provided)
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        # Add JSON output instruction
        if output_schema:
            schema_instruction = "\n\nIMPORTANT: You must return ONLY valid JSON that conforms to the specified schema. Do not include any markdown formatting, code blocks, or explanatory text."
            user_message = user_message + schema_instruction
            messages[1]["content"] = user_message
        
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=False
                )
                
                content = response.choices[0].message.content.strip()
                
                # Try to extract JSON if wrapped in markdown
                if "```json" in content:
                    start = content.find("```json") + 7
                    end = content.find("```", start)
                    content = content[start:end].strip()
                elif "```" in content:
                    start = content.find("```") + 3
                    end = content.find("```", start)
                    content = content[start:end].strip()
                
                # Parse JSON
                parsed = None
                try:
                    parsed = json.loads(content)
                except json.JSONDecodeError as e:
                    # Try to fix truncated JSON
                    fixed_content = self._try_fix_truncated_json(content)
                    if fixed_content != content:
                        try:
                            parsed = json.loads(fixed_content)
                        except json.JSONDecodeError:
                            parsed = None  # Fix failed, will retry
                    
                    if parsed is None:
                        if attempt < self.max_retries - 1:
                            time.sleep(self.retry_delay * (attempt + 1))
                            continue
                        # Show more context in error
                        error_msg = f"Failed to parse JSON response: {e}\nResponse length: {len(content)} chars\nResponse preview: {content[:1000]}"
                        if len(content) > 1000:
                            error_msg += f"\n... (truncated, showing last 500 chars) ...\n{content[-500:]}"
                        raise ValueError(error_msg)
                
                # Validate against schema if provided
                if output_schema:
                    try:
                        validated = output_schema(**parsed)
                        return validated.model_dump(mode='json')
                    except ValidationError as e:
                        if attempt < self.max_retries - 1:
                            time.sleep(self.retry_delay * (attempt + 1))
                            continue
                        raise ValueError(f"Schema validation failed: {e}\nParsed: {parsed}")
                
                return parsed
                
            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue
                raise RuntimeError(f"LLM API request failed after {self.max_retries} attempts: {e}")
        
        raise RuntimeError("Failed to get valid response from LLM")
    
    def _try_fix_truncated_json(self, content: str) -> str:
        """Try to fix truncated/partial JSON by trimming to the last balanced point"""
        stripped = content.lstrip()
        if not stripped.startswith('{') and not stripped.startswith('['):
            return content

        # Walk the string, track brace/brace stack while skipping quoted text
        stack = []
        in_string = False
        escape_next = False
        last_balanced = -1
        for i, ch in enumerate(content):
            if escape_next:
                escape_next = False
                continue
            if ch == '\\':
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch in '{[':
                stack.append(ch)
            elif ch in '}]':
                if stack:
                    stack.pop()
                else:
                    # Unmatched closing, stop tracking
                    break
            if not stack:
                last_balanced = i

        # If we ever reached a balanced point, trim to there; otherwise keep original
        fixed = content if last_balanced == -1 else content[:last_balanced + 1]
        fixed = fixed.rstrip()

        # Drop a trailing comma that would keep the JSON invalid
        while fixed and fixed[-1] in {',', ' ', '\n', '\t', '\r'}:
            if fixed[-1] == ',':
                fixed = fixed[:-1]
                break
            fixed = fixed[:-1]

        # Close any remaining open delimiters in reverse order
        closing = {'{': '}', '[': ']'}
        for opener in reversed(stack):
            fixed += closing.get(opener, '')

        return fixed
    
    def chat_completion_with_schema_description(
        self,
        system_prompt: str,
        user_message: str,
        schema_description: str,
        temperature: float = 0.3,
        max_tokens: int = 4000
    ) -> Dict[str, Any]:
        """
        Send chat completion with schema description (for complex schemas)
        
        Args:
            system_prompt: System prompt
            user_message: User message
            schema_description: JSON schema description as string
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        
        Returns:
            Parsed JSON response
        """
        schema_instruction = f"\n\nOutput Format:\n{schema_description}\n\nReturn ONLY valid JSON matching this schema."
        full_user_message = user_message + schema_instruction
        
        return self.chat_completion(
            system_prompt=system_prompt,
            user_message=full_user_message,
            output_schema=None,
            temperature=temperature,
            max_tokens=max_tokens
        )
