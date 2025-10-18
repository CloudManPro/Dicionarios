import json
import boto3
import os
import urllib3
import re
import base64
import traceback # Import traceback for printing exception details

# ==============================================================================
# Configuration
# ==============================================================================

# AWS Environment Variables
# Ensure REGION and AWS_SSM_PARAMETER_TARGET_NAME_0 are set in Lambda Environment
region = os.getenv("REGION", "us-east-1")
parameter_name = os.getenv("AWS_SSM_PARAMETER_TARGET_NAME_0")

# Validate essential environment variables
if not parameter_name:
    print("CRITICAL: FATAL: Environment variable AWS_SSM_PARAMETER_TARGET_NAME_0 is not set.")
    # In a real Lambda, raising an error might be better if config is essential
    # raise ValueError("Missing required environment variable: AWS_SSM_PARAMETER_TARGET_NAME_0")
if not region:
    print("WARN: Environment variable REGION not set, defaulting to us-east-1.")

# ==============================================================================
# AWS Client Initialization
# ==============================================================================
ssm = None # Initialize to None
try:
    if region: # Proceed only if region is set
        ssm = boto3.client('ssm', region_name=region)
        print(f"INFO: SSM client initialized for region: {region}")
    else:
         print("ERROR: Cannot initialize SSM client because REGION is not set.")
except Exception as e:
    print(f"CRITICAL: FATAL: Failed to initialize Boto3 SSM client for region {region}: {e}")
    # Depending on requirements, might raise error or handle in handler
    # raise e

# ==============================================================================
# Global Variables / Constants
# ==============================================================================
GEMINI_API_KEY = None  # Cached API Key

# Gemini Model/API Configuration
# Using a 1.5 model likely requires Beta API and supports grounding/multimodal
DEFAULT_GEMINI_MODEL = "gemini-1.5-flash-latest"
BETA_API_VERSION = "v1beta"
DEFAULT_API_VERSION = BETA_API_VERSION # Default to Beta for features

print(f"INFO: Default Gemini Model: {DEFAULT_GEMINI_MODEL}, Default API Version: {DEFAULT_API_VERSION}")

# Custom Separator for SyntesysAI Citations
CITATION_SEPARATOR = "###CITATIONS###"

# ==============================================================================
# Helper Functions
# ==============================================================================

def get_gemini_api_key():
    """Retrieves and caches the Gemini API Key from SSM Parameter Store."""
    global GEMINI_API_KEY
    if GEMINI_API_KEY:
        print("INFO: Using cached Gemini API Key.")
        return GEMINI_API_KEY

    if not parameter_name:
        print("ERROR: SSM Parameter name environment variable is not configured.")
        raise ValueError("SSM Parameter name is not configured.")
    if not ssm:
        print("ERROR: SSM client is not initialized (failed or region missing). Cannot retrieve API Key.")
        raise ValueError("SSM client failed to initialize.")

    try:
        print(f"INFO: Attempting to retrieve API Key from SSM Parameter: {parameter_name}")
        ssm_response = ssm.get_parameter(Name=parameter_name, WithDecryption=True)
        GEMINI_API_KEY = ssm_response['Parameter']['Value']
        if not GEMINI_API_KEY:
             print("ERROR: Retrieved API Key from SSM is empty.")
             raise ValueError("Retrieved API Key from SSM is empty.")
        print(f"INFO: API Key retrieved successfully (first 4 chars): {GEMINI_API_KEY[:4]}...")
        return GEMINI_API_KEY
    except ssm.exceptions.ParameterNotFound:
        print(f"CRITICAL: FATAL: SSM Parameter '{parameter_name}' not found in region '{region}'. Check parameter name and region.")
        traceback.print_exc()
        raise ValueError(f"SSM Parameter '{parameter_name}' not found.") # Raise specific error
    except Exception as e:
        print(f"CRITICAL: FATAL: Error retrieving API key from SSM: {e}")
        traceback.print_exc()
        raise ConnectionError(f"Failed to retrieve API key from SSM: {e}") # Raise specific error

def create_marker(text, num_words=3):
    """Creates a 'first N words ... last N words' marker from text."""
    if not text or not isinstance(text, str):
        return ""
    words = re.findall(r'\b\w+\b', text)
    if len(words) == 0:
        return ""
    if len(words) <= 2 * num_words:
        return ' '.join(words)
    else:
        start_words = " ".join(words[:num_words])
        end_words = " ".join(words[-num_words:])
        return f"{start_words}...{end_words}"

def make_gemini_request(api_key, api_version, model_name, payload_contents, generation_config=None, tools=None, timeout=120.0): # Increased timeout slightly
    """Sends a request to the Gemini API and returns the HTTP response object."""
    endpoint_action = "generateContent"
    gemini_api_url = f"https://generativelanguage.googleapis.com/{api_version}/models/{model_name}:{endpoint_action}?key={api_key}"
    print(f"INFO: Making request to Gemini API URL: {gemini_api_url.split('?key=')[0]}?key=...") # Hide key in URL log

    headers = {"Content-Type": "application/json"}
    payload = {"contents": payload_contents}

    if generation_config:
        payload["generationConfig"] = generation_config
    if tools:
        payload["tools"] = tools

    # Log payload safely (redact large data)
    try:
         log_payload = json.loads(json.dumps(payload)) # Deep copy
         # --- Redaction Logic ---
         if 'contents' in log_payload and isinstance(log_payload['contents'], list):
             for content_item in log_payload['contents']:
                 if isinstance(content_item, dict) and 'parts' in content_item and isinstance(content_item['parts'], list):
                     for part in content_item['parts']:
                         if isinstance(part, dict):
                             if 'text' in part and isinstance(part['text'], str) and len(part['text']) > 500:
                                 part['text'] = part['text'][:250] + f"... (truncated {len(part['text'])} chars)"
                             if 'inline_data' in part and isinstance(part['inline_data'], dict) and 'data' in part['inline_data']:
                                  data_len = len(part['inline_data']['data'])
                                  part['inline_data']['data'] = f"<base64 data ({data_len} bytes)>"

         print(f"DEBUG: Payload sent to Gemini (redacted): {json.dumps(log_payload, ensure_ascii=False, indent=2)}")
    except Exception as log_e:
         print(f"WARN: Could not fully redact payload for printing: {log_e}")
         print(f"DEBUG: Payload structure keys: {list(payload.keys()) if isinstance(payload, dict) else 'Not a dict'}")

    http = urllib3.PoolManager()
    try:
        response = http.request(
            "POST",
            gemini_api_url,
            body=json.dumps(payload).encode("utf-8"),
            headers=headers,
            timeout=timeout
        )
        print(f"DEBUG: Gemini response status: {response.status}")
        return response
    except urllib3.exceptions.ReadTimeoutError:
        print(f"ERROR: Gemini API call timed out after {timeout} seconds.")
        traceback.print_exc() # Optional: Print traceback for timeout
        raise TimeoutError(f"Gemini API timeout after {timeout}s")
    except Exception as e:
        print(f"ERROR: Error during Gemini API call: {e}")
        traceback.print_exc()
        raise ConnectionError(f"Gemini API request failed: {e}")

def create_proxy_response(status_code, body_object):
    """Creates a standard AWS Lambda Proxy response dictionary with CORS."""
    try:
        # Ensure the body_object itself is serializable before dumping
        # A simple check, might need more complex validation depending on structure
        json.dumps(body_object)
        body_string = json.dumps(body_object, ensure_ascii=False)
    except TypeError as e:
        print(f"ERROR: Failed to serialize response body object: {e}. Object type: {type(body_object)}")
        # Attempt to serialize a fallback error message
        fallback_body = {"error": "Internal server error: Failed to serialize response.", "details": str(e)}
        try:
            body_string = json.dumps(fallback_body, ensure_ascii=False)
            status_code = 500 # Override status code to indicate internal error
        except Exception as fallback_e:
             print(f"CRITICAL: FATAL: Could not even serialize fallback error message: {fallback_e}")
             body_string = '{"error": "Internal server error: Unserializable response and error."}'
             status_code = 500

    return {
        "statusCode": status_code,
        "headers": {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Allow-Methods": "OPTIONS,POST"
        },
        "body": body_string
    }

# ==============================================================================
# Main Lambda Handler
# ==============================================================================

def lambda_handler(event, context):
    """
    Main Lambda handler: parses input, calls Gemini, processes response, returns proxy format.
    """
    action = None # Initialize action
    input_data = {} # Initialize input_data

    try:
        # Attempt to get API key early - fail fast if config issue
        api_key = get_gemini_api_key()

        request_id = getattr(context, 'aws_request_id', 'N/A')
        print(f"DEBUG: Lambda execution started. Request ID: {request_id}")
        # Limit event logging in production if it contains sensitive data
        # print(f"DEBUG: Received raw event: {json.dumps(event)}") # Log serialized event

        # --- Input parsing logic ---
        if isinstance(event, dict):
            # Check if action is directly in the event (e.g., test event)
            if 'action' in event:
                print("INFO: Reading parameters directly from event object.")
                input_data = event
            # Check if it's an API Gateway proxy event with a body
            elif 'body' in event:
                print("INFO: Reading parameters from event['body'].")
                body_content = event.get('body')
                if isinstance(body_content, str):
                    try:
                        input_data = json.loads(body_content)
                    except json.JSONDecodeError:
                        print("ERROR: Invalid JSON in event['body'].")
                        return create_proxy_response(400, {"error": "Invalid JSON format in request body."})
                elif isinstance(body_content, dict):
                    # Body might already be parsed if using certain integrations/middlewares
                    input_data = body_content
                else:
                     print(f"WARN: event['body'] has unexpected type: {type(body_content).__name__}. Treating as empty.")
                     input_data = {}
            else:
                print("WARN: Event is a dict but lacks 'action' and 'body'. Assuming direct input.")
                input_data = event # Treat as direct input if unsure

        else:
            # Handle non-dict events if necessary, otherwise return error
            print(f"ERROR: Received event is not a dictionary: {type(event).__name__}")
            return create_proxy_response(400, {"error": "Invalid event format."})

        # Extract action from the parsed input_data
        action = input_data.get("action")
        if not action:
             print("ERROR: Missing 'action' parameter in request payload.")
             return create_proxy_response(400, {"error": "O parâmetro 'action' é obrigatório na requisição."})

        print(f"INFO: Processing action: {action}")

        # Initialize common variables
        payload_contents = []
        final_payload = {} # Payload to be returned in the proxy response body
        tokens_info = {}
        api_version = DEFAULT_API_VERSION # Default, may be overridden by action
        model_name = DEFAULT_GEMINI_MODEL # Default, currently not overridden
        generation_config = None # Example: {"temperature": 0.7, "maxOutputTokens": 8192}
        tools = None # Example: [{"grounding": {...}}] - VERIFY DOCS

        # --- Action-Specific Logic ---

        if action == "SyntesysAI":
            api_version = BETA_API_VERSION # Grounding often needs Beta
            user_query = input_data.get("user_query")
            source_documents = input_data.get("source_documents") # Expecting list of {title, content}

            if not user_query or not isinstance(user_query, str):
                print("ERROR: 'user_query' missing or invalid for SyntesysAI.")
                return create_proxy_response(400, {"error": "'user_query' (string) obrigatório para 'SyntesysAI'."})
            if not source_documents or not isinstance(source_documents, list) or len(source_documents) == 0:
                 print("ERROR: 'source_documents' missing or invalid for SyntesysAI.")
                 return create_proxy_response(400, {"error": "'source_documents' (lista não vazia de objetos com 'title' e 'content') obrigatório para 'SyntesysAI'."})

            # --- Construct Payload for Grounding (Multi-Turn) ---
            content_parts = [] # MUST BE INITIALIZED HERE
            doc_map = {} # Map index back to title for potential API attribution processing
            print(f"DEBUG: Processing {len(source_documents)} source documents for grounding.")

            for i, doc in enumerate(source_documents):
                if not isinstance(doc, dict):
                    print(f"WARN: Item in source_documents at index {i} is not a dictionary, skipping.")
                    continue
                title = doc.get("title", f"Documento {i+1}")
                content = doc.get("content", "")
                if not content or not isinstance(content, str) or not content.strip():
                    print(f"WARN: Document '{title}' (index {i}) has invalid/empty content, skipping.")
                    continue

                doc_map[i] = title
                # Using clear separators helps the model distinguish documents.
                content_parts.append({"text": f"---\nDocument Title: {title}\n---\n{content}\n---\nEnd Document: {title}\n---"})
                print(f"DEBUG: Added document index {i}: '{title}' (len: {len(content)})")

            # Check if any valid documents were processed
            if not content_parts: # Check AFTER the loop
                 print("ERROR: No documents with valid content found to send for grounding.")
                 return create_proxy_response(400, {"error": "Nenhum documento com conteúdo válido fornecido."})

            # Define the text for the final user query part
            query_part_text = (
                #"INSTRUÇÃO IMPORTANTE: Responda à seguinte pergunta utilizando EXCLUSIVAMENTE "
                #"as informações contidas nos documentos fornecidos nos turnos anteriores. Não use conhecimento externo.\n\n"
                #"INSTRUÇÃO IMPORTANTE: dê preferencia ao conteúdos fornecidos para a resposta e com isso amplie usando sua base de conhecimento."
                f"PERGUNTA DO USUÁRIO: {user_query}\n\n"
                "Cite os trechos relevantes dos textos usados na resposta, cada um precedido por EXATAMENTE a sequência 'Trecho:<titulo do texto>:'."
                "em seguida aos trechos, faça um resumo do que vc entendeu."
                # Instruction for formatting the output with separator and titles
                f"APÓS fornecer a resposta completa à pergunta acima, adicione uma linha contendo EXATAMENTE a sequência '{CITATION_SEPARATOR}'.\n"
                "DEPOIS dessa linha de separação, liste APENAS os títulos dos documentos (fornecidos em 'Document Title:') que você REALMENTE utilizou para formular sua resposta, um título por linha. Não liste documentos que não foram usados."
            )
            query_part = {"text": query_part_text}

            # Construct Multi-Turn Payload list
            print("INFO: Constructing multi-turn payload for Gemini with explicit citation formatting instruction.")
            turn1_user_docs = {"role": "user", "parts": content_parts}
            turn2_model_ack = {"role": "model", "parts": [{"text": "Ok, recebi os documentos fornecidos. Por favor, faça sua pergunta baseada neles e siga as instruções de formatação da resposta."}]}
            turn3_user_query = {"role": "user", "parts": [query_part]}
            payload_contents = [turn1_user_docs, turn2_model_ack, turn3_user_query]

            # Potentially add grounding tools if required by the API version/model (VERIFY DOCS)
            # Example (Hypothetical):
            # tools = [{"grounding": {"source": {"content": content_parts}}}] # Or other structure

            print(f"INFO: Prepared multi-turn payload for grounding with {len(content_parts)} documents and formatting instruction.")

        elif action == "KeyWords":
            keyWords_input = input_data.get("keyWords")
            if not keyWords_input or not isinstance(keyWords_input, str):
                print("ERROR: 'keyWords' missing or invalid for KeyWords.")
                return create_proxy_response(400, {"error": "'keyWords' (string) obrigatório para 'KeyWords'."})
            # Adjust prompt as needed for keyword extraction
            combined_prompt = f"Extraia as palavras-chave mais relevantes do seguinte texto: '{keyWords_input}'\nListe apenas as palavras-chave, separadas por vírgula."
            payload_contents = [{"role": "user", "parts": [{"text": combined_prompt}]}]
            api_version = DEFAULT_API_VERSION # Use default (Beta might be fine)

        elif action == "FixText":
            text_input = input_data.get("text")
            if not text_input or not isinstance(text_input, str):
                print("ERROR: 'text' missing or invalid for FixText.")
                return create_proxy_response(400, {"error": "'text' (string) obrigatório para 'FixText'."})
            # Adjust prompt as needed for text correction
            combined_prompt = f"Corrija a gramática e a ortografia do texto a seguir, mantendo o significado original:\n\n{text_input}"
            payload_contents = [{"role": "user", "parts": [{"text": combined_prompt}]}]
            api_version = DEFAULT_API_VERSION # Use default (Beta might be fine)

        elif action in ["convertHTML", "convertTXT"]:
             api_version = BETA_API_VERSION # Multimodal requires Beta
             pdf_base64 = input_data.get("pdf_base64")
             if not pdf_base64 or not isinstance(pdf_base64, str):
                  print(f"ERROR: 'pdf_base64' missing or invalid for {action}.")
                  return create_proxy_response(400, {"error": f"'pdf_base64' (string) obrigatório para '{action}'."})

             if action == "convertHTML":
                 # Use the detailed prompt from original code, ensure clarity
                 prompt_input = (
                     "Converta o conteúdo do PDF anexo em um único arquivo HTML semanticamente estruturado. "
                     "Use tags HTML apropriadas (h1-h6, p, ul, ol, li, table, etc.) para representar a estrutura do documento (títulos, parágrafos, listas, tabelas). "
                     "Preserve a formatação básica como negrito e itálico, se possível, usando tags <b>/<strong> e <i>/<em>. "
                     "Não inclua CSS inline extenso, foque na estrutura semântica. Ignore cabeçalhos e rodapés repetitivos. "
                     "Sua resposta DEVE ser APENAS um objeto JSON válido contendo uma única chave: "
                     "'html_content': Uma string contendo APENAS o código HTML completo resultante."
                 )
             else: # action == "convertTXT"
                 # Use the detailed prompt from original code, ensure clarity
                 prompt_input = (
                     "Extraia o texto puro do conteúdo do PDF anexo, preservando a ordem e as quebras de linha originais o máximo possível. "
                     "Instruções importantes:\n"
                     "1- Preserve toda quebra de linha original do documento.\n"
                     "2- Remova cabeçalhos e rodapés que se repetem em múltiplas páginas (como números de página ou títulos constantes).\n"
                     "3- Não adicione nenhuma formatação ou texto que não esteja no documento original.\n"
                     "4- Tente lidar corretamente com hifenização no final das linhas, juntando as palavras quando apropriado.\n"
                     "Sua resposta DEVE ser APENAS um objeto JSON válido contendo uma única chave: "
                     "'txt_content': Uma string contendo APENAS o texto puro extraído."
                 )

             # Construct multimodal payload
             payload_contents = [ { "role": "user", "parts": [ {"text": prompt_input}, {"inline_data": {"mime_type": "application/pdf", "data": pdf_base64}} ] } ]
             print(f"INFO: Using Gemini API {api_version} for multimodal action '{action}'.")

        else:
            print(f"ERROR: Ação '{action}' não suportada.")
            return create_proxy_response(400, {"error": f"Ação '{action}' não suportada."})

        # --- Call Gemini API ---
        # Ensure payload_contents was actually assigned by one of the actions
        if not payload_contents:
             print(f"CRITICAL: Internal logic error: payload_contents not set for action '{action}'.")
             return create_proxy_response(500, {"error": "Erro interno do servidor: Falha na preparação da requisição."})

        gemini_response = make_gemini_request(
            api_key, api_version, model_name, payload_contents,
            generation_config=generation_config, tools=tools
        )

        # --- Process Gemini Response ---
        response_status = gemini_response.status
        try:
            response_data = gemini_response.data.decode("utf-8")
        except Exception as decode_err:
             print(f"ERROR: Failed to decode Gemini response data (Status {response_status}): {decode_err}")
             return create_proxy_response(500, {"error": "Erro interno: Falha ao decodificar resposta da IA."})

        print(f"DEBUG: Gemini API Status Code: {response_status}")
        # Limit printing raw response data in production for security/verbosity
        print(f"DEBUG: Gemini Raw Response (start): {response_data[:500]}{'...' if len(response_data) > 500 else ''}")

        if response_status != 200:
            error_details = response_data
            try:
                error_json = json.loads(response_data)
                error_details = error_json.get("error", {}).get("message", response_data)
                print(f"DEBUG: Parsed Gemini Error JSON: {error_json}")
            except json.JSONDecodeError:
                print("WARN: Gemini error response was not valid JSON.")
            except Exception as e:
                print(f"WARN: Could not parse Gemini error JSON: {e}")

            print(f"ERROR: Erro da API Gemini ({response_status}): {error_details}")
            # Map Gemini status codes if needed, otherwise use 502 for bad gateway
            proxy_status = response_status if response_status >= 400 else 502
            return create_proxy_response(proxy_status, {"error": f"Erro de comunicação com a IA ({response_status})", "details": error_details})

        # --- Parse Successful Response JSON ---
        try:
            result_json = json.loads(response_data)
            # Avoid logging full successful response unless necessary for deep debug
            # print(f"DEBUG: Parsed Gemini Response JSON: {json.dumps(result_json, indent=2, ensure_ascii=False)}")
        except json.JSONDecodeError as e:
            print(f"ERROR: Falha ao decodificar JSON da resposta Gemini BEM SUCEDIDA (Status {response_status}): {e}")
            traceback.print_exc()
            print(f"ERROR: Resposta bruta que falhou no parse: {response_data}")
            return create_proxy_response(500, {"error": "Erro interno: Resposta da IA não era JSON válido."})

        # --- Validate Gemini response structure & check safety ---
        if p_feedback := result_json.get("promptFeedback"):
             # Check only for BLOCKING reasons
             if reason := p_feedback.get("blockReason"):
                 print(f"WARN: Gemini prompt BLOCKED: {reason}. Details: {p_feedback.get('blockReasonMessage','')}. Ratings: {p_feedback.get('safetyRatings')}")
                 return create_proxy_response(400, {"error": f"Requisição bloqueada pela IA (prompt): {reason}", "details": p_feedback.get('blockReasonMessage','')})
             else:
                  print(f"DEBUG: Prompt Feedback (non-blocking): {p_feedback}")

        candidates = result_json.get('candidates')
        if not candidates or not isinstance(candidates, list) or len(candidates) == 0:
             print(f"ERROR: Resposta Gemini sem 'candidates' válidos: {result_json}")
             return create_proxy_response(500, {"error": "Erro interno: Estrutura de resposta inesperada da IA (sem candidatos)."})

        candidate = candidates[0] # Assume first candidate is the primary one

        # Check finish reason and safety rating of the candidate
        finish_reason = candidate.get("finishReason")
        if finish_reason and finish_reason not in ["STOP", "MAX_TOKENS"]:
             print(f"WARN: Candidate finished with non-standard reason: {finish_reason}")
             if finish_reason == "SAFETY":
                 ratings = candidate.get("safetyRatings")
                 print(f"WARN: Gemini response candidate BLOCKED: SAFETY. Ratings: {ratings}")
                 blocked_category = "Unknown"
                 if ratings:
                     for rating in ratings:
                         # Check for categories other than HARM_SEVERITY_NEGLIGIBLE or similar safe ratings if available
                         # Simplified: block if any rating exists and finishReason is SAFETY
                         blocked_category = rating.get("category", "Unknown")
                         break # Block on first listed category for simplicity
                 return create_proxy_response(400, {"error": f"Resposta bloqueada pela IA (segurança): {blocked_category}"})
             elif finish_reason == "RECITATION":
                  print(f"WARN: Gemini response candidate finished due to RECITATION.")
                  # May need specific handling depending on requirements
             # Handle other reasons like "OTHER" if necessary

        content = candidate.get('content')
        if not content or not isinstance(content, dict) or not (parts := content.get('parts')) or not isinstance(parts, list) or len(parts) == 0:
            print(f"ERROR: Resposta Gemini com estrutura de 'content'/'parts' inválida: {candidate}")
            return create_proxy_response(500, {"error": "Erro interno: Estrutura de conteúdo inesperada da IA."})

        # Check if the first part actually contains text
        if 'text' not in parts[0] or not isinstance(parts[0]['text'], str):
             print(f"ERROR: Primeira parte da resposta Gemini não contém 'text' válido: {parts[0]}")
             # If it finished normally but has no text, it's an issue
             if finish_reason == "STOP":
                 return create_proxy_response(500, {"error": "Erro interno: IA retornou conteúdo sem texto."})
             else: # If it finished for other reasons, might be expected (e.g., function call planned for future)
                  print(f"WARN: Primeira parte da resposta sem texto, mas finishReason foi '{finish_reason}'. Retornando payload vazio (sem 'answer').")
                  gemini_output_text = "" # Treat as empty text if no text part
                  # Allow processing to continue to extract tokens etc.
             # return create_proxy_response(500, {"error": f"Erro interno: IA terminou sem texto ({finish_reason})."})

        else:
            gemini_output_text = parts[0]['text'] # The main generated text

        # --- Construct Final Payload based on Action ---
        final_payload = {} # Reset just in case

        # Handle standard text/JSON actions
        if action in ["convertHTML", "convertTXT"]:
            print(f"DEBUG: Processing '{action}'. Expecting JSON in Gemini text.")
            cleaned_text = gemini_output_text.strip()
            # Regex to find JSON block, potentially wrapped in ```json ... ```
            match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', cleaned_text, re.DOTALL | re.IGNORECASE)
            json_string_to_parse = match.group(1) if match else cleaned_text
            if match: print("DEBUG: Markdown fences found and JSON extracted.")
            else: print("DEBUG: No Markdown fences found, attempting to parse entire text as JSON.")
            try:
                parsed_gemini_output = json.loads(json_string_to_parse)
                if not isinstance(parsed_gemini_output, dict):
                     raise ValueError("Parsed content is not a JSON object (dictionary).")

                key_to_extract = "html_content" if action == "convertHTML" else "txt_content"
                fallback_key = "html" if action == "convertHTML" else "text"

                content_value = parsed_gemini_output.get(key_to_extract) or parsed_gemini_output.get(fallback_key)

                if content_value is None or not isinstance(content_value, str):
                    print(f"ERROR: Parsed JSON missing/invalid '{key_to_extract}'/'{fallback_key}'. Keys: {list(parsed_gemini_output.keys())}")
                    raise ValueError(f"Estrutura JSON da IA incompleta/inesperada para {action}.")

                final_payload[key_to_extract] = content_value # Use the correct key for the response
                print(f"INFO: Conversão para '{action}' processada com sucesso.")

            except (json.JSONDecodeError, ValueError) as e:
                print(f"ERROR: Erro ao processar/validar JSON interno da Gemini para '{action}': {e}")
                traceback.print_exc()
                print(f"ERROR: String JSON (após limpeza) que falhou: {json_string_to_parse[:1000]}{'...' if len(json_string_to_parse) > 1000 else ''}")
                # Return the raw text if JSON parsing fails? Or always error? Let's error for now.
                return create_proxy_response(500, {"error": f"Erro interno: Formato de dados inválido da IA para '{action}'.", "details": str(e)})

        elif action in ["KeyWords", "FixText"]:
             processed_response = gemini_output_text.strip()
             final_payload["response"] = processed_response # Use a generic "response" key
             print(f"INFO: Resposta simples processada para '{action}'.")

        # --- Handle SyntesysAI with Grounding / Separator ---
        elif action == "SyntesysAI":
            print("INFO: Processing SyntesysAI response with custom separator logic.")
            processed_answer = ""
            extracted_titles = []
            api_citations = [] # Store citations potentially from API grounding

            # 1. Try parsing using the custom separator
            parts = gemini_output_text.split(CITATION_SEPARATOR)
            if len(parts) > 1:
                processed_answer = parts[0].strip()
                titles_block = parts[1].strip()
                extracted_titles = [title.strip() for title in titles_block.split('\n') if title.strip()]
                if extracted_titles:
                    print(f"INFO: Separator '{CITATION_SEPARATOR}' found. Extracted {len(extracted_titles)} titles.")
                    final_payload["citations_extracted"] = extracted_titles # Use a distinct key
                else:
                     print(f"WARN: Separator '{CITATION_SEPARATOR}' found, but no valid titles listed after it.")
                # Use the text before the separator as the main answer
                final_payload["answer"] = processed_answer
            else:
                # Separator not found, use the whole text as the answer
                print(f"WARN: Separator '{CITATION_SEPARATOR}' not found in response. Using full text as answer.")
                final_payload["answer"] = gemini_output_text.strip()

            # 2. (Optional/Fallback) Try processing API grounding attributions
            # ** VERIFY THE ACTUAL KEY IN GEMINI DOCS ('groundingAttributions', 'citationMetadata', etc.) **
            attribution_key = 'groundingAttributions' # ASSUMPTION
            attributions = candidate.get(attribution_key)
            print(f"DEBUG: Checking for native API attributions under key '{attribution_key}'. Found type: {type(attributions)}")

            if attributions and isinstance(attributions, list):
                print(f"INFO: Found {len(attributions)} potential native API grounding attributions.")
                processed_segments = set()
                for i, attr in enumerate(attributions):
                    segment_text = None
                    source_index = None
                    # Adapt extraction logic based on actual API response structure
                    # Example checks (may need adjustment):
                    if isinstance(attr.get("content"), dict) and isinstance(attr["content"].get("textSegment"), str): segment_text = attr["content"]["textSegment"]
                    elif isinstance(attr.get("segment"), str): segment_text = attr["segment"]
                    if isinstance(attr.get("sourceId"), dict) and isinstance(attr["sourceId"].get("partIndex"), int): source_index = attr["sourceId"]["partIndex"]
                    elif isinstance(attr.get("documentIndex"), int): source_index = attr["documentIndex"]

                    if segment_text and source_index is not None and source_index in doc_map:
                        source_title = doc_map[source_index]
                        if segment_text not in processed_segments:
                             marker = create_marker(segment_text)
                             api_citations.append({ "source_title": source_title, "marker": marker, "segment": segment_text })
                             processed_segments.add(segment_text)
                             print(f"DEBUG: Processed native citation: Title='{source_title}', Marker='{marker}'")
                        else:
                             print(f"DEBUG: Skipping duplicate native segment for Title='{source_title}'")
                    else:
                        print(f"WARN: Could not process native attribution {i}: Segment={segment_text is not None}, Index={source_index}, Mapped={doc_map.get(source_index)}")

                if api_citations:
                    print(f"INFO: Successfully processed {len(api_citations)} native API citations.")
                    final_payload["citations_api"] = api_citations # Use a distinct key
                elif not extracted_titles: # Only warn if separator also failed
                     print("WARN: Native attributions list found, but no valid citations could be processed from it.")

            elif not extracted_titles: # Only warn if separator also failed
                print(f"WARN: No native grounding attributions found under key '{attribution_key}' or format is invalid.")

            # Final check: ensure answer key exists even if empty
            if "answer" not in final_payload: final_payload["answer"] = ""
            if not extracted_titles and not api_citations:
                 print("INFO: No citations extracted via separator or native API grounding.")
                 final_payload["citations_extracted"] = [] # Ensure key exists
                 final_payload["citations_api"] = [] # Ensure key exists

            print(f"INFO: SyntesysAI processed. Answer length: {len(final_payload.get('answer',''))}, Extracted Titles: {len(extracted_titles)}, API Citations: {len(api_citations)}")


        # --- Extract token usage metadata (Common to all successful calls) ---
        usage_metadata = result_json.get("usageMetadata", {})
        prompt_tokens = usage_metadata.get("promptTokenCount", 0)
        candidates_tokens = usage_metadata.get("candidatesTokenCount", 0)
        total_tokens = usage_metadata.get("totalTokenCount", 0)
        # Calculate total if not provided by API
        if total_tokens == 0 and (prompt_tokens > 0 or candidates_tokens > 0):
             total_tokens = prompt_tokens + candidates_tokens
             print("DEBUG: Calculated totalTokenCount as sum of prompt and candidates.")
        tokens_info = {
            "promptTokenCount": prompt_tokens,
            "candidatesTokenCount": candidates_tokens,
            "totalTokenCount": total_tokens
        }
        final_payload["tokens"] = tokens_info
        print(f"INFO: Token Usage: {json.dumps(tokens_info)}")

        # --- Return Success ---
        print(f"INFO: Action '{action}' completed successfully.")
        return create_proxy_response(200, final_payload) # Return the structured final payload

    # --- Exception Handling ---
    except (TimeoutError, ConnectionError) as api_err:
        print(f"ERROR: API Call Error: {api_err}")
        # traceback.print_exc() # Optional traceback for network errors
        status = 504 if isinstance(api_err, TimeoutError) else 502
        error_detail = str(api_err)
        return create_proxy_response(status, {"error": f"Erro de comunicação com a IA.", "details": error_detail})
    except ValueError as ve:
        # Catch specific config/value errors handled earlier or during processing
        print(f"ERROR: Data validation or configuration error: {ve}")
        traceback.print_exc()
        # Determine if it's a config error (missing env var, SSM issue) or bad request data
        is_config_error = "SSM" in str(ve) or "environment variable" in str(ve)
        status_code = 500 if is_config_error else 400
        error_msg = "Erro de configuração do servidor" if is_config_error else "Erro nos dados da requisição ou processamento"
        return create_proxy_response(status_code, {"error": error_msg, "details": str(ve)})
    except Exception as e:
        # Catchall for unexpected errors during processing
        current_action = action if action else "Initialization/Parsing"
        print(f"CRITICAL: Unhandled exception processing action '{current_action}': {type(e).__name__} - {e}")
        traceback.print_exc() # Print full traceback for unexpected errors
        return create_proxy_response(500, {"error": "Erro interno inesperado do servidor.", "details": f"{type(e).__name__}: {e}"})

# Example Usage (for local testing, replace with actual event/context)
if __name__ == '__main__':
    # --- Mock Event for SyntesysAI ---
    mock_event_synthesys = {
      "action": "SyntesysAI",
      "user_query": "Resuma os tipos de alta mencionados.",
      "source_documents": [
        {"title": "Doc 1 Title", "content": "Conteúdo do documento 1 falando sobre alta tipo A."},
        {"title": "Doc 2 Title", "content": "Conteúdo do documento 2 falando sobre alta tipo B e mencionando tipo A."},
        {"title": "Doc 3 Title (Empty)", "content": "  "},
        {"title": "Doc 4 Title", "content": "Conteúdo do documento 4 falando sobre alta tipo C."}
      ]
    }
    # --- Mock Event for convertHTML ---
    mock_event_html = {
        "action": "convertHTML",
        "pdf_base64": "JVBERi0xLjcKCjEgMCBvYmogICUgZW50cnkgcG9pbnQKPDwKICAvVHlwZSAvQ2F0YWxvZwogIC9QYWdlcyAyIDAgUgo+PgplbmRvYmoKCjIgMCBvYmoKPDwKICAvVHlwZSAvUGFnZXMKICAvTWVkaWFCb3ggWzAgMCAyMDAgMjAwXQogIC9Db3VudCAxCiAgL0tpZHMgWyAzIDAgUiBdCj4+CmVuZG9iagoKMyAwIG9iago8PAogIC9UeXBlIC9QYWdlCiAgL1BhcmVudCAyIDAgUgogIC9SZXNvdXJjZXMKPDwKICAgIC9Gb250Cjw8CiAgICAgIC9GMQo8PAogICAgICAgIC9UeXBlIC9Gb250CiAgICAgICAgL1N1YnR5cGUgL1R5cGUxCiAgICAgICAgL0Jhc2VGb250IC9UaW1lcy1Sb21hbgogICAgICA+PgoKICAgID4+CiAgPC9Qcm9jU2V0IFsvUERGLC9UZXh0L0ltYWdlQi9JbWFnZUMvSW1hZ2VJXQogID4+CiAgL0NvbnRlbnRzIDQgMCBSID4+CmVuZG9iagoKNC"+ base64.b64encode(b'%PDF-1.7\n1 0 obj<</Type/Page/Parent 2 0 R/Contents 4 0 R/MediaBox[0 0 612 792]/Resources<</Font<</F1 5 0 R>> >> >>endobj\n4 0 obj<</Length 35>>stream\nBT /F1 12 Tf 72 720 Td (Hello PDF!)Tj ET\nendstream\nendobj\n5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n2 0 obj<</Type/Pages/Count 1/Kids[1 0 R]>>endobj\n6 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\ntrailer<</Size 7/Root 6 0 R>>\n%%EOF').decode('ascii') # Minimal valid PDF base64
    }

    # --- Mock Context ---
    class MockContext:
        aws_request_id = "local-test-123"
        # Add other context attributes if needed

    mock_context = MockContext()

    print("\n--- Testing SyntesysAI ---")
    # Make sure AWS_SSM_PARAMETER_TARGET_NAME_0 env var is set or mock get_gemini_api_key
    # For local test, you might hardcode the key or use another method
    # os.environ['AWS_SSM_PARAMETER_TARGET_NAME_0'] = 'your-ssm-parameter-name'
    # os.environ['REGION'] = 'your-region'
    try:
        # You might need to mock boto3 ssm calls for local testing without AWS creds/config
        # Or temporarily replace get_gemini_api_key() call inside lambda_handler
        # with a hardcoded key for local test run ONLY.
        # Example: api_key = "YOUR_ACTUAL_API_KEY_FOR_TESTING"
        # Make sure to remove hardcoded keys before deployment.
        print("Local test run: API key retrieval from SSM might fail if not configured.")
        response_synthesys = lambda_handler(mock_event_synthesys, mock_context)
        print("\nSyntesysAI Response:")
        print(json.dumps(json.loads(response_synthesys['body']), indent=2, ensure_ascii=False))
        print(f"Status Code: {response_synthesys['statusCode']}")
    except Exception as e:
        print(f"\nSyntesysAI Test FAILED: {e}")
        traceback.print_exc()

