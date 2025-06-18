import replicate
import os
from dotenv import load_dotenv

load_dotenv()
replicate_client = replicate.Client(api_token=os.getenv('REPLICATE_API_TOKEN'))

model = replicate_client.models.get('black-forest-labs/flux-kontext-max')
print(model)

