import yaml
import json
import os

openapi_file_path = "/home/yueqis/agent/docprompting/data/openapi_v2.yaml"
with open(openapi_file_path, 'r') as file:
    openapi_data = yaml.safe_load(file)

def find_api_name_and_method(target_line):
    # Traverse through the paths section of the OpenAPI definition
    paths = openapi_data['paths']
    outputs = set()
    for path, methods in paths.items():
        # Traverse through the HTTP methods (e.g., get, post, put, delete) under each path
        for method, details in methods.items():
            # Check if the target line is in the operation summary or description
            if 'summary' in details and 'summary: ' in target_line and target_line.split('summary: ')[1] in details['summary']:
                outputs.add(f"{method} {path}")
            if 'description' in details and 'description: ' in target_line and target_line.split('description: ')[1] in details['description']:
                outputs.add(f"{method} {path}")
            if ('/api/v4' in target_line or '/api/v3' in target_line) and path in target_line:
                outputs.add(f"{method} {path}")
    return outputs

def parse_retrieval_result(task_id, retrieval_result_file):
    task_id = str(task_id)
    with open(retrieval_result_file, 'r') as file:
        retrieval_result = json.load(file)
    task_result = retrieval_result[task_id]['retrieved']
    outputs = set()
    for target_line in task_result:
        outputs = outputs | find_api_name_and_method(target_line)
    return outputs

# get yaml of the selected api
def get_yaml_by_api(apis, output_file_name):
    projects = {}
    defns = {}
    yaml_data = openapi_data
    for api in apis:
        method = api.split(" ")[0]
        call = api.split(" ")[1]
        try:
            path = yaml_data['paths'][call][method]
            projects[call] = {method: path}
            # get refs
            if 'responses' not in path: continue
            if '200' not in path['responses']: continue
            if ('schema' in path['responses']['200']):
                schema = path['responses']['200']['schema']
                if ('items' in schema): ref = os.path.basename(schema['items']['$ref'])
                else: 
                    if '$ref' in schema: ref = os.path.basename(schema['$ref'])
                    else: continue
                defn = yaml_data['definitions'][ref]
                defns[ref] = defn
        except: continue
    with open(output_file_name, 'w') as file:
        yaml.dump({'host': 'gitlab.com', 'paths': projects, 'definitions': defns}, file, default_flow_style=False, sort_keys=False)

task_ids = [132, 133, 134, 135, 136, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177]
for task_id in task_ids:
    outputs = parse_retrieval_result(task_id, "data/retrieval_result.json")
    print(len(outputs))
    get_yaml_by_api(list(outputs), f"api/{task_id}.yaml")
