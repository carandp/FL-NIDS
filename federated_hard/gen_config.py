import json
import os
import shutil

template_path = "jobs/nids_fedavg/app/config/config_fed_client.template.json"
output_path = "jobs/nids_fedavg/app/config/config_fed_client.json"

with open(template_path) as f:
    content = f.read().replace("<username>", os.environ["USER"])

with open(output_path, "w") as f:
    f.write(content)

print(f"✅ config_fed_client.json generated for user: {os.environ['USER']}")