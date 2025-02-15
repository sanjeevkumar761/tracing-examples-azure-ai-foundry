# Install the following dependencies: azure.identity and azure-ai-inference
import os
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.identity import DefaultAzureCredential
from azure.core.credentials import AzureKeyCredential

from opentelemetry import trace

# Install opentelemetry with command "pip install opentelemetry-sdk".
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter
from azure.core.settings import settings
from azure.ai.inference.tracing import AIInferenceInstrumentor
#from azure.monitor.opentelemetry import configure_azure_monitor
from dotenv import load_dotenv

load_dotenv()

settings.tracing_implementation = "opentelemetry"


# Requires opentelemetry-sdk
span_exporter = ConsoleSpanExporter()
tracer_provider = TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))
trace.set_tracer_provider(tracer_provider)

# [START trace_function]
from opentelemetry.trace import get_tracer

tracer = get_tracer(__name__)

endpoint = os.getenv("AZURE_INFERENCE_SDK_ENDPOINT")
key = os.getenv("AZURE_INFERENCE_SDK_KEY")
client = ChatCompletionsClient(endpoint=endpoint, credential=AzureKeyCredential(key))

@tracer.start_as_current_span("get_chat_response")
def get_chat_response():
    span = trace.get_current_span()
    span.set_attribute("model", "gpt-4o")
    response = client.complete(
        messages=[
            SystemMessage(content="You are a helpful assistant."),
            UserMessage(content="What are 3 things to visit in Seattle?")
        ],
        max_tokens=1000
    )
    print(response)

AIInferenceInstrumentor().instrument()

get_chat_response()

AIInferenceInstrumentor().uninstrument()