# -*- coding: utf-8 -*-
# DBAccess.py (or lambda_function.py)

import boto3
import os
import base64
import json
import traceback
import time
import re
import unicodedata # Using native normalization
from botocore.exceptions import ClientError

# --- MODULE IMPORT ---
try:
    import InsertDynamo # For atomic operations
    print("[INFO] Successfully imported InsertDynamo module.")
except ImportError:
    print("[ERROR] FAILED TO IMPORT InsertDynamo module. Actions 'upDateDB', 'deleteDB', 'createJSON', and atomic 'deleteItem' DB ops WILL FAIL.")
    InsertDynamo = None

# --- CONSTANTS ---
PARTITION_KEY_NAME = os.environ.get('DYNAMODB_PARTITION_KEY_NAME', 'Partition')
SORT_KEY_NAME = os.environ.get('DYNAMODB_SORT_KEY_NAME', 'Sort')
PDF_PREFIX = "pdf/"
HTML_PREFIX = "html/"
DEFAULT_COMPOSITE_KEY_SEPARATOR = ':' # Needed if InsertDynamo uses it and doesn't define its own default

# --- ENVIRONMENT VARIABLES ---
DYNAMODB_TABLE_NAME = os.environ.get("AWS_DYNAMODB_TABLE_TARGET_NAME_0")
S3_BUCKET_NAME = os.environ.get("AWS_S3_BUCKET_TARGET_NAME_0")
AWS_REGION = os.environ.get("REGION")

# --- Defaults for List Item ---
DEFAULT_LISTA_Publication_PK_VALUE = "#listaPublication"
DEFAULT_LIST_ATTRIBUTE_NAME = "TitulosSet"
LIST_PK_VALUE = os.environ.get('DYNAMODB_LIST_PK_VALUE', DEFAULT_LISTA_Publication_PK_VALUE)
LIST_ATTR_NAME = os.environ.get('DYNAMODB_LIST_ATTRIBUTE_NAME', DEFAULT_LIST_ATTRIBUTE_NAME)

# --- BOTO3 CLIENTS/RESOURCES ---
dynamodb_resource = None
dynamodb_client = None
s3_client = None
try:
    if AWS_REGION:
        print(f"[INFO] Initializing Boto3 clients for region: {AWS_REGION}")
        boto_session = boto3.Session(region_name=AWS_REGION)
    else:
        print("[WARN] AWS_REGION environment variable not set. Using default region configuration.")
        boto_session = boto3.Session()

    dynamodb_resource = boto_session.resource('dynamodb')
    dynamodb_client = boto_session.client('dynamodb')
    s3_client = boto_session.client('s3')
    print("[INFO] Boto3 clients initialized successfully.")
except Exception as e:
    print(f"[ERROR] Failed to initialize Boto3 clients: {e}")
    raise RuntimeError(f"Boto3 client initialization failed: {e}")

# --- Helper Function to Convert Sets ---
def convert_sets(obj):
    if isinstance(obj, set): return list(obj)
    if isinstance(obj, dict): return {k: convert_sets(v) for k, v in obj.items()}
    if isinstance(obj, list): return [convert_sets(item) for item in obj]
    return obj

# --- NATIVE PYTHON ACCENT REMOVAL & NORMALIZATION ---
def normalize_s3_key_component(component_str):
    """ Normalizes a string for S3 key using only native Python libraries. """
    if not isinstance(component_str, str):
        print(f"[WARN] Normalization: Input is not a string: {type(component_str)} '{component_str}'. Returning empty string.")
        return ""
    original_for_log = component_str
    try:
        nfkd_form = unicodedata.normalize('NFD', original_for_log)
        normalized = "".join([c for c in nfkd_form if not unicodedata.category(c) == 'Mn'])
        normalized = normalized.lower()
        normalized = re.sub(r'[\s_.:;,/\\\'"!?()\[\]{}]+', '_', normalized)
        normalized = re.sub(r'[^a-z0-9_-]+', '', normalized)
        normalized = re.sub(r'[_]+', '_', normalized)
        normalized = re.sub(r'[-]+', '-', normalized)
        normalized = normalized.strip('_-')
        if not normalized: normalized = 'invalid_component'; print(f"[WARN] Normalization resulted in empty string for input '{original_for_log}', using fallback.")
    except Exception as e:
        print(f"[ERROR] Normalization: Error during NFD normalization/filtering on '{original_for_log}': {e}. Falling back.")
        # Simple fallback if unicode fails
        normalized = original_for_log.lower()
        normalized = re.sub(r'[^a-z0-9_-]+', '_', normalized) # Replace invalid with _
        normalized = re.sub(r'[_]+', '_', normalized).strip('_')
        if not normalized: normalized = 'invalid_component_fallback'

    print(f"[INFO] Normalization Final Result (Native): Input='{original_for_log}', Output='{normalized}'")
    return normalized

# --- S3 Upload Logic ---
def handle_s3_upload(payload, s3_bucket_name):
    # (Function unchanged from previous version)
    print(f"[INFO] Starting handle_s3_upload for bucket: {s3_bucket_name}")
    object_key = payload.get("objectKey"); file_content_base64 = payload.get("body"); file_name = payload.get("fileName"); content_type_from_payload = payload.get("contentType", 'application/octet-stream')
    if not object_key: raise ValueError("S3 Input Error: 'objectKey' missing.")
    if not file_content_base64 or not isinstance(file_content_base64, str): raise ValueError("S3 Input Error: Base64 content 'body' missing or invalid.")
    if not file_name:
        print(f"[INFO] S3 Info: Deriving fileName from objectKey: '{object_key}'")
        try: file_name = os.path.basename(object_key) or f"unknown_file_{int(time.time())}"
        except Exception as e: file_name = f"unknown_file_{int(time.time())}"; print(f"[WARN] S3 Warning: Error deriving filename: {e}.")
    print(f"[INFO] S3 Params: fileName='{file_name}', objectKey='{object_key}', contentType='{content_type_from_payload}'")
    try:
        missing_padding = len(file_content_base64) % 4;
        if missing_padding: file_content_base64 += '=' * (4 - missing_padding)
        file_content_bytes = base64.b64decode(file_content_base64, validate=True)
    except (base64.binascii.Error, ValueError, TypeError) as e: raise ValueError(f"S3 Input Error: Invalid Base64 encoding: {str(e)}")
    except Exception as e: raise RuntimeError(f"S3 Server Error during file decoding: {str(e)}")
    try:
        if not s3_client: raise RuntimeError("S3 Client not initialized.")
        s3_client.put_object(Bucket=s3_bucket_name, Key=object_key, Body=file_content_bytes, ContentType=content_type_from_payload)
        print(f"[INFO] S3 Success: File '{file_name}' uploaded to '{object_key}'.")
        return { "message": f"File '{file_name}' uploaded successfully.", "bucket": s3_bucket_name, "objectKey": object_key, "derivedFileName": file_name }
    except ClientError as e: error_code = e.response.get('Error', {}).get('Code', 'Unknown'); error_message = e.response.get('Error', {}).get('Message', str(e)); print(f"[ERROR] S3 Upload ClientError: {error_code} - {error_message}"); raise e
    except Exception as e: print(f"[ERROR] S3 Upload Unexpected Error: {str(e)}"); print(traceback.format_exc()); raise RuntimeError(f"S3 Unexpected Server Error during upload: {str(e)}")

# --- S3 Content Deletion Logic ---
def handle_s3_content_deletion(payload, s3_bucket_name):
    # (Function unchanged from previous version - uses normalize_s3_key_component defined above)
    print(f"[INFO] Starting handle_s3_content_deletion for bucket: {s3_bucket_name}")
    results = {"pdf_deleted": False, "html_deleted": False, "messages": []}
    partition_key_value = payload.get("partition"); sort_key_value = payload.get("sort") # Expecting Original Title/Pub here
    if not partition_key_value: msg = "S3 Deletion Error: 'partition' key (Original Pub Name) missing."; print(f"[ERROR] {msg}"); results["messages"].append(msg); return results
    if not sort_key_value: msg = "S3 Deletion Error: 'sort' key (Original Title) missing."; print(f"[ERROR] {msg}"); results["messages"].append(msg); return results
    if not s3_client: msg = "S3 Deletion Error: S3 Client not initialized."; print(f"[ERROR] {msg}"); results["messages"].append(msg); return results
    if not s3_bucket_name: msg = "S3 Deletion Error: S3 Bucket Name not configured."; print(f"[ERROR] {msg}"); results["messages"].append(msg); return results
    try:
        print(f"[DEBUG] S3 Deletion: Normalizing Partition Key: '{partition_key_value}'")
        norm_partition = normalize_s3_key_component(partition_key_value)
        print(f"[DEBUG] S3 Deletion: Normalizing Sort Key: '{sort_key_value}'")
        norm_sort = normalize_s3_key_component(sort_key_value)
    except Exception as norm_error: msg = f"S3 Deletion Error: Failed during key normalization: {norm_error}"; print(f"[ERROR] {msg}"); results["messages"].append(msg); return results
    if not norm_partition or not norm_sort or norm_partition == 'invalid_component' or norm_sort == 'invalid_component': msg = f"S3 Deletion Error: Normalization resulted in invalid/empty component(s). NormPart='{norm_partition}', NormSort='{norm_sort}'"; print(f"[ERROR] {msg}"); results["messages"].append(msg); return results
    pdf_key = f"{PDF_PREFIX}{norm_partition}/{norm_sort}.pdf"; html_key = f"{HTML_PREFIX}{norm_partition}/{norm_sort}.html"
    keys_to_delete = [(pdf_key, "PDF"), (html_key, "HTML")]
    for s3_key, file_type in keys_to_delete:
        print(f"[INFO] S3 Deletion: Attempting to delete {file_type} file with key: s3://{s3_bucket_name}/{s3_key}")
        try:
            response = s3_client.delete_object(Bucket=s3_bucket_name, Key=s3_key)
            status_code = response.get('ResponseMetadata', {}).get('HTTPStatusCode')
            if status_code == 204: msg = f"S3 Deletion Success: {file_type} file '{s3_key}' deleted (or did not exist)."; print(f"[INFO] {msg}"); results[f"{file_type.lower()}_deleted"] = True; results["messages"].append(msg)
            else: msg = f"S3 Deletion Warning: Received status {status_code} for key '{s3_key}'. Response: {response}"; print(f"[WARN] {msg}"); results["messages"].append(msg)
        except ClientError as e: error_code = e.response.get('Error', {}).get('Code', 'Unknown'); error_message = e.response.get('Error', {}).get('Message', str(e)); msg = f"S3 Deletion ClientError deleting {file_type} key '{s3_key}': {error_code} - {error_message}"; print(f"[ERROR] {msg}"); results["messages"].append(msg)
        except Exception as e: msg = f"S3 Deletion Unexpected Error deleting {file_type} key '{s3_key}': {str(e)}"; print(f"[ERROR] {msg}"); print(traceback.format_exc()); results["messages"].append(msg)
    print(f"[INFO] S3 Deletion Process completed for PK='{partition_key_value}', SK='{sort_key_value}'. Results: {results}"); return results


# --- DynamoDB Get Item ---
def get_dynamodb_item(table_name, partition_key_value, sort_key_value):
    # (Function unchanged)
    if partition_key_value is None or sort_key_value is None: raise ValueError("GetItem Error: Both partition and sort key values are required.")
    if not dynamodb_resource: raise RuntimeError("DynamoDB Resource not initialized.")
    dynamodb_table = dynamodb_resource.Table(table_name); key = { PARTITION_KEY_NAME: partition_key_value, SORT_KEY_NAME: sort_key_value }
    print(f"[INFO] GetItem: Attempting to get item with Key: {key} from table: {table_name}")
    try:
        response = dynamodb_table.get_item(Key=key)
        if 'Item' in response: print("[INFO] GetItem: Item found."); return response['Item']
        else: print("[INFO] GetItem: Item not found."); return None
    except ClientError as e: error_code = e.response.get("Error", {}).get("Code"); error_message = e.response.get("Error", {}).get("Message", str(e)); print(f"[ERROR] GetItem ClientError: [{error_code}] {error_message}"); raise e
    except Exception as e: print(f"[ERROR] GetItem Unexpected Error: {str(e)}"); print(traceback.format_exc()); raise RuntimeError(f"GetItem Unexpected Server Error: {str(e)}")

# --- DynamoDB Put Item ---
def put_dynamodb_item(table_name, partition_key_value, sort_key_value, item_data):
    # (Function unchanged)
    if partition_key_value is None or sort_key_value is None: raise ValueError("PutItem Error: Both partition and sort key values are required.")
    if not isinstance(item_data, dict) or not item_data: raise ValueError("PutItem Error: item_data must be a non-empty dictionary.")
    if PARTITION_KEY_NAME not in item_data: raise ValueError(f"PutItem Error: item_data dictionary is missing the partition key '{PARTITION_KEY_NAME}'.")
    if SORT_KEY_NAME not in item_data: raise ValueError(f"PutItem Error: item_data dictionary is missing the sort key '{SORT_KEY_NAME}'.")
    item_data_pk_val = str(item_data[PARTITION_KEY_NAME]); item_data_sk_val = str(item_data[SORT_KEY_NAME]); provided_pk_val = str(partition_key_value); provided_sk_val = str(sort_key_value)
    if item_data_pk_val != provided_pk_val: raise ValueError(f"PutItem Error: Mismatch partition key ('{item_data_pk_val}' vs '{provided_pk_val}')")
    if item_data_sk_val != provided_sk_val: raise ValueError(f"PutItem Error: Mismatch sort key ('{item_data_sk_val}' vs '{provided_sk_val}')")
    if not dynamodb_resource: raise RuntimeError("DynamoDB Resource not initialized.")
    dynamodb_table = dynamodb_resource.Table(table_name); key_info = { PARTITION_KEY_NAME: partition_key_value, SORT_KEY_NAME: sort_key_value }
    print(f"[INFO] PutItem: Attempting to put item with Key: {key_info} into table: {table_name}")
    try:
        response = dynamodb_table.put_item(Item=item_data); status_code = response.get('ResponseMetadata', {}).get('HTTPStatusCode')
        if status_code == 200: print(f"[INFO] PutItem: Item successfully put. Status Code: {status_code}"); return response.get('ResponseMetadata')
        else: print(f"[WARN] PutItem Warning: Put operation completed but received non-200 status: {status_code}"); print(f"[WARN] Full Response: {response}"); return response.get('ResponseMetadata')
    except ClientError as e: error_code = e.response.get("Error", {}).get("Code"); error_message = e.response.get("Error", {}).get("Message", str(e)); print(f"[ERROR] PutItem ClientError: [{error_code}] {error_message} for Key: {key_info}"); raise e
    except Exception as e: print(f"[ERROR] PutItem Unexpected Error: {str(e)} for Key: {key_info}"); print(traceback.format_exc()); raise RuntimeError(f"PutItem Unexpected Server Error: {str(e)}")

# --- Simple DynamoDB Delete Item (NO LONGER USED BY 'deleteItem' ACTION) ---
# Keep it for potential other uses? Or remove if definitely unused.
# def delete_dynamodb_item(table_name, partition_key_value, sort_key_value):
#     if partition_key_value is None or sort_key_value is None: raise ValueError("DeleteItem Error: Both partition and sort key values are required.")
#     if not dynamodb_resource: raise RuntimeError("DynamoDB Resource not initialized.")
#     dynamodb_table = dynamodb_resource.Table(table_name); key = { PARTITION_KEY_NAME: partition_key_value, SORT_KEY_NAME: sort_key_value }
#     print(f"[INFO] (Simple) DeleteItem: Attempting to delete item with Key: {key} from table: {table_name}")
#     # ... rest of simple delete logic ...

# --- DynamoDB Query ---
def get_dynamodb_items_by_partition_query(table_name, partition_key_value):
    # (Function unchanged)
    if partition_key_value is None: raise ValueError("Query Error: Partition key value is missing for query.")
    if not dynamodb_resource: raise RuntimeError("DynamoDB Resource not initialized.")
    partition_key_name = PARTITION_KEY_NAME; all_items = []; last_evaluated_key = None; dynamodb_table = dynamodb_resource.Table(table_name)
    print(f"[INFO] Query: Starting query on table '{table_name}' for {partition_key_name}='{partition_key_value}'")
    try:
        while True:
            query_kwargs = { 'KeyConditionExpression': f'#pk = :pkval', 'ExpressionAttributeNames': {'#pk': partition_key_name}, 'ExpressionAttributeValues': {':pkval': partition_key_value} }
            if last_evaluated_key: query_kwargs['ExclusiveStartKey'] = last_evaluated_key
            response = dynamodb_table.query(**query_kwargs); items = response.get('Items', []); all_items.extend(items); last_evaluated_key = response.get('LastEvaluatedKey')
            if not last_evaluated_key: break
        print(f"[INFO] Query: Found {len(all_items)} item(s) for partition key '{partition_key_value}'.")
        return all_items
    except ClientError as e: error_code = e.response.get("Error", {}).get("Code"); error_message = e.response.get("Error", {}).get("Message", str(e)); print(f"[ERROR] Query ClientError querying table {table_name} for PK '{partition_key_value}': [{error_code}] {error_message}"); raise e
    except Exception as e: print(f"[ERROR] Query Unexpected Error querying table {table_name} for PK '{partition_key_value}': {e}"); print(traceback.format_exc()); raise RuntimeError(f"Query Unexpected Server Error: {str(e)}")


# === AWS LAMBDA HANDLER ===
def lambda_handler(event, context):
    print(f"Lambda Handler started. Event: {json.dumps(event)}")
    start_time = time.time(); action = None; result_data = None; final_message = "Action processing started."
    if dynamodb_resource is None or dynamodb_client is None or s3_client is None: raise RuntimeError("Lambda dependencies (Boto3 clients) failed initialization.")
    if not DYNAMODB_TABLE_NAME: raise EnvironmentError("Config Error: Missing DYNAMODB_TABLE_NAME env var.")

    try:
        print(f"[INFO] Using Table: '{DYNAMODB_TABLE_NAME}'"); print(f"[INFO] Using Bucket: '{S3_BUCKET_NAME if S3_BUCKET_NAME else 'Not Configured'}'"); print(f"[INFO] Using List PK: '{LIST_PK_VALUE}', List Attr: '{LIST_ATTR_NAME}'"); print(f"[INFO] Using PK Name: '{PARTITION_KEY_NAME}', SK Name: '{SORT_KEY_NAME}'")
        if not isinstance(event, dict): raise ValueError("Invalid request: Expected JSON object payload.")
        payload = event; action = payload.get("access"); print(f"[INFO] Action received: {action}")
        if not action: raise ValueError("Invalid request: 'access' key missing in payload.")

        # --- Routing ---
        if action == "upDateDB":
            print("[INFO] Routing to InsertDynamo.handle_dynamodb_update_atomic...")
            if InsertDynamo is None: raise RuntimeError("InsertDynamo module not loaded.")
            result_data = InsertDynamo.handle_dynamodb_update_atomic( payload=payload, dynamodb_table_name=DYNAMODB_TABLE_NAME, dynamodb_client=dynamodb_client, partition_key_name=PARTITION_KEY_NAME, sort_key_name=SORT_KEY_NAME, list_pk_value=LIST_PK_VALUE, list_attr_name=LIST_ATTR_NAME ); final_message = "DynamoDB atomic update processed."
        elif action == "uploadFile":
             if not S3_BUCKET_NAME: raise EnvironmentError("Config Error: Missing S3_BUCKET_NAME for uploadFile.")
             print("[INFO] Routing to handle_s3_upload..."); result_data = handle_s3_upload(payload, S3_BUCKET_NAME); final_message = "S3 upload processed."
        elif action == "getItem":
            print("[INFO] Routing to get_dynamodb_item..."); partition = payload.get("partition"); sort = payload.get("sort")
            result_data = get_dynamodb_item(DYNAMODB_TABLE_NAME, partition, sort)
            if result_data is None: print("[INFO] getItem: Item not found."); final_message = "Item not found."; return None
            else: final_message = "Item retrieved successfully."
        elif action == "putItem":
            print("[INFO] Routing to put_dynamodb_item..."); partition = payload.get("partition"); sort = payload.get("sort"); item_to_put = payload.get("item")
            if not item_to_put: raise ValueError("PutItem Error: 'item' data missing in payload.")
            print(f"[INFO] Calling put_dynamodb_item for PK='{partition}', SK='{sort}'...")
            response_metadata = put_dynamodb_item( table_name=DYNAMODB_TABLE_NAME, partition_key_value=partition, sort_key_value=sort, item_data=item_to_put ); print("[INFO] putItem action finished."); final_message = "Item put successfully."; result_data = response_metadata

        # --- UPDATED 'deleteItem' ACTION ---
        elif action == "deleteItem":
            print("[INFO] Routing to ATOMIC deleteItem (DB Transaction + S3)...")
            # We need the payload from FE containing original values for indices
            if InsertDynamo is None:
                raise RuntimeError("InsertDynamo module not loaded, cannot perform atomic DB removal.")

            # --- ADJUSTED VALIDATION: Data is no longer required here ---
            if not payload.get("Titulo") or not payload.get("Nome_da_Publication"):
                 raise ValueError("Atomic Deletion Error: Payload must include original Titulo and Nome_da_Publication.")
            # --- END ADJUSTED VALIDATION ---

            # Step 1: Perform the ATOMIC DynamoDB removal (main item, indexes, list update)
            # handle_dynamodb_removal_atomic will now fetch the Date/SK internally
            db_results = {}
            db_removal_success = False
            db_message = "DynamoDB atomic removal step skipped or failed."
            try:
                print("[INFO] deleteItem: Attempting DynamoDB atomic removal via InsertDynamo module...")
                # Pass the full payload, the atomic function will use what it needs
                db_results = InsertDynamo.handle_dynamodb_removal_atomic(
                    payload=payload, # Pass the full payload (Titulo, Nome_pub, Esp, Aut, Cat1, Cat2, Tipo...)
                    dynamodb_table_name=DYNAMODB_TABLE_NAME,
                    dynamodb_client=dynamodb_client,
                    partition_key_name=PARTITION_KEY_NAME,
                    sort_key_name=SORT_KEY_NAME,
                    list_pk_value=LIST_PK_VALUE,
                    list_attr_name=LIST_ATTR_NAME,
                    separator=DEFAULT_COMPOSITE_KEY_SEPARATOR
                )
                # Check the status returned by the atomic function if added
                if db_results.get("status") == "NOT_FOUND":
                     db_removal_success = False # Or True depending on how you want to treat non-existence
                     db_message = db_results.get("message", "Main item not found in DynamoDB.")
                     # Decide if you want to proceed with S3 delete even if DB item wasn't found
                     # proceed_with_s3 = False # Example: Stop S3 if DB item absent
                elif db_results.get("status") == "DELETED": # Assuming status field was added
                     db_removal_success = True
                     db_message = db_results.get("message", "DynamoDB atomic removal transaction completed successfully.")
                else: # Fallback if status field wasn't added/returned
                     db_removal_success = True # Assume success if no exception
                     db_message = db_results.get("message", "DynamoDB atomic removal transaction completed successfully.")

                print(f"[INFO] deleteItem: DynamoDB atomic removal attempt finished. Success reported: {db_removal_success}")

            except (ClientError, ValueError, RuntimeError) as db_error:
                print(f"[ERROR] deleteItem: CRITICAL failure during DynamoDB atomic removal: {db_error}. Aborting S3 deletion.")
                db_message = f"DynamoDB atomic removal FAILED: {db_error}"
                raise db_error # Re-raise to stop processing and report failure
            except Exception as db_exc:
                print(f"[ERROR] deleteItem: UNEXPECTED CRITICAL failure during DynamoDB atomic removal: {db_exc}.")
                print(traceback.format_exc())
                db_message = f"DynamoDB atomic removal FAILED unexpectedly: {db_exc}"
                raise db_exc # Re-raise

            # Step 2: Attempt S3 deletion (Only if DB step indicated potential success or item was found)
            s3_results = {}
            s3_message = "S3 deletion skipped."
            # Optional: Add check `and db_results.get("status") != "NOT_FOUND"` if you don't want to delete S3 for non-existent DB items
            if S3_BUCKET_NAME:
                 try:
                     # S3 deletion still needs original Titulo and Nome_da_Publication
                     s3_payload = {
                         "partition": payload.get("Nome_da_Publication"),
                         "sort": payload.get("Titulo")
                     }
                     if s3_payload["partition"] and s3_payload["sort"]: # Ensure keys are present
                         s3_results = handle_s3_content_deletion(s3_payload, S3_BUCKET_NAME)
                         s3_message = "S3 deletion attempt finished."
                         print(f"[INFO] deleteItem: {s3_message} Results: {s3_results}")
                     else:
                         s3_message = "S3 deletion skipped (missing original keys in payload)."
                         print(f"[WARN] deleteItem: {s3_message}")
                         s3_results = {"messages": [s3_message]}
                 except Exception as s3_exc:
                      print(f"[ERROR] deleteItem: Unexpected error during S3 content deletion call: {s3_exc}")
                      s3_results = {"messages": [f"Unexpected error during S3 deletion: {s3_exc}"]}
                      s3_message = f"S3 deletion attempt FAILED unexpectedly: {s3_exc}"
            elif not S3_BUCKET_NAME:
                 s3_message = "S3 deletion skipped (bucket not configured)."
                 print(f"[WARN] deleteItem: {s3_message}")
                 s3_results = {"messages": [s3_message]}
            # Step 3: Consolidate result message
            combined_messages = [db_message, s3_message] + s3_results.get("messages", [])
            # Remove potential duplicates or overly verbose messages if needed
            final_message = "DeleteItem action finished. Summary:\n- " + "\n- ".join(list(dict.fromkeys(combined_messages))) # Basic dedupe
            print(f"[INFO] {final_message}")
            # Return a consolidated success message including S3 details
            result_data = {
                "message": final_message,
                # Use the more specific flag from the atomic function result if available
                "dynamoDbAtomicRemovalSuccess": db_results.get("status") == "DELETED" if "status" in db_results else db_removal_success,
                "s3PdfDeleted": s3_results.get("pdf_deleted", False),
                "s3HtmlDeleted": s3_results.get("html_deleted", False)
            }
            if "transactionItemsAttempted" in db_results:
                result_data["dynamoDbItemsTargeted"] = db_results["transactionItemsAttempted"]
            if "status" in db_results: # Pass along status like NOT_FOUND if needed
                result_data["dynamoDbStatus"] = db_results["status"]


        elif action == "queryItem":
            print("[INFO] Routing to get_dynamodb_items_by_partition_query..."); partition = payload.get("partition")
            result_data = get_dynamodb_items_by_partition_query(DYNAMODB_TABLE_NAME, partition)
            item_count = len(result_data) if isinstance(result_data, list) else 'N/A'; final_message = f"Query completed for partition '{partition}'. Found {item_count} items."
        elif action == "deleteDB":
            print("[INFO] Routing to InsertDynamo.delete_Publication_items_atomic...")
            if InsertDynamo is None: raise RuntimeError("InsertDynamo module not loaded.")
            result_data = InsertDynamo.delete_Publication_items_atomic( payload=payload, dynamodb_table_name=DYNAMODB_TABLE_NAME, dynamodb_client=dynamodb_client, partition_key_name=PARTITION_KEY_NAME, sort_key_name=SORT_KEY_NAME, list_pk_value=LIST_PK_VALUE, list_attr_name=LIST_ATTR_NAME ); final_message = "Atomic publication delete processed."
        
        elif action == "createJSON":
            print("[INFO] Routing to InsertDynamo.create_json_from_publication...")
            if InsertDynamo is None: raise RuntimeError("InsertDynamo module not loaded.")
            publication_name = payload.get("nome_publication");
            if not publication_name: raise ValueError("CreateJSON Error: 'nome_publication' missing.")
            result_data = InsertDynamo.create_json_from_publication( publication_name=publication_name, dynamodb_table_name=DYNAMODB_TABLE_NAME, dynamodb_client=dynamodb_client, s3_client=s3_client, s3_bucket_name=S3_BUCKET_NAME, partition_key_name=PARTITION_KEY_NAME, sort_key_name=SORT_KEY_NAME, list_pk_value=LIST_PK_VALUE, list_attr_name=LIST_ATTR_NAME ); final_message = "JSON creation process finished."
        else: raise ValueError(f"Invalid request: Unknown 'access' action provided: '{action}'")

        # --- Return ---
        processing_time = round((time.time() - start_time) * 1000); print(f"[INFO] Lambda Handler completed successfully for action '{action}'. Final Message: '{final_message}'. Time: {processing_time}ms.")
        if result_data is None and action == 'getItem': pass
        elif isinstance(result_data, dict):
            if 'message' not in result_data and final_message: result_data['message'] = final_message
            result_data['processingTimeMs'] = processing_time
        elif isinstance(result_data, list): result_data = { "items": result_data, 'message': final_message, 'processingTimeMs': processing_time }
        elif result_data is None: result_data = { 'message': final_message or f'Action {action} completed, no data returned.', 'processingTimeMs': processing_time }
        if result_data is not None: result_data = convert_sets(result_data)
        return result_data

    # --- Error Handling ---
    except (ValueError, EnvironmentError, RuntimeError, ClientError) as e:
        processing_time = round((time.time() - start_time) * 1000); error_type = type(e).__name__; print(f"[ERROR] Lambda Handler FAILED for action '{action}'. Error: [{error_type}] {str(e)}. Time: {processing_time}ms.")
        if not isinstance(e, (ValueError, EnvironmentError)): print(traceback.format_exc()); raise
    except Exception as e:
        processing_time = round((time.time() - start_time) * 1000); error_type = type(e).__name__; print(f"[ERROR] Lambda Handler FAILED with UNEXPECTED error for action '{action}'. Error: [{error_type}] {str(e)}. Time: {processing_time}ms.")
        print(traceback.format_exc()); raise RuntimeError(f"Unexpected Server Error during action '{action}': {str(e)}") from e