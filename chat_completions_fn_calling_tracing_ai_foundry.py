
import os
from opentelemetry import trace

# Install opentelemetry with command "pip install opentelemetry-sdk".
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage, CompletionsFinishReason
from azure.core.credentials import AzureKeyCredential
from azure.monitor.opentelemetry.exporter import AzureMonitorTraceExporter
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from azure.core.settings import settings
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient

from opentelemetry.trace import get_tracer

from dotenv import load_dotenv

load_dotenv()

project_client = AIProjectClient.from_connection_string(
    credential=DefaultAzureCredential(),
    conn_str="eastus.api.azureml.ms;24304329-da71-44a3-b653-7fcc08964744;newazureai;sanjeku-0128",
)

# [START trace_setting]
settings.tracing_implementation = "opentelemetry"
span_exporter = ConsoleSpanExporter()
tracer_provider = TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))
trace.set_tracer_provider(tracer_provider)
# [END trace_setting]


tracer = get_tracer(__name__)
exporter = AzureMonitorTraceExporter(
    connection_string=project_client.telemetry.get_connection_string()
)
span_processor = BatchSpanProcessor(exporter)
trace.get_tracer_provider().add_span_processor(span_processor)

# [START trace_function]
# The tracer.start_as_current_span decorator will trace the function call and enable adding additional attributes
# to the span in the function implementation. Note that this will trace the function parameters and their values.
@tracer.start_as_current_span("get_temperature")  # type: ignore
def get_temperature(city: str) -> str:

    # Adding attributes to the current span
    span = trace.get_current_span()
    span.set_attribute("requested_city", city)

    if city == "Seattle":
        return "75"
    elif city == "New York City":
        return "80"
    else:
        return "Unavailable"


# [END trace_function]

@tracer.start_as_current_span("get_weather")
def get_weather(city: str) -> str:
    span = trace.get_current_span()
    span.set_attribute("requested_city", city)

    if city == "Seattle":
        return "Nice weather"
    elif city == "New York City":
        return "Good weather"
    else:
        return "Unavailable"


def chat_completion_with_function_call(key, endpoint):
    import json
    from azure.ai.inference.models import (
        ToolMessage,
        AssistantMessage,
        ChatCompletionsToolCall,
        ChatCompletionsToolDefinition,
        FunctionDefinition,
    )

    weather_description = ChatCompletionsToolDefinition(
        function=FunctionDefinition(
            name="get_weather",
            description="Returns description of the weather in the specified city",
            parameters={
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The name of the city for which weather info is requested",
                    },
                },
                "required": ["city"],
            },
        )
    )

    temperature_in_city = ChatCompletionsToolDefinition(
        function=FunctionDefinition(
            name="get_temperature",
            description="Returns the current temperature for the specified city",
            parameters={
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The name of the city for which temperature info is requested",
                    },
                },
                "required": ["city"],
            },
        )
    )

    client = ChatCompletionsClient(
        endpoint=endpoint, credential=AzureKeyCredential(key),

        )
    messages = [
        SystemMessage("You are a helpful assistant."),
        UserMessage("What is the weather and temperature in Seattle?"),
    ]


    response = client.complete(
        messages=messages, 
        tools=[weather_description, temperature_in_city],
        #model = "gpt-4o",
        
    )

    if response.choices[0].finish_reason == CompletionsFinishReason.TOOL_CALLS:
        # Append the previous model response to the chat history
        messages.append(AssistantMessage(tool_calls=response.choices[0].message.tool_calls))
        # The tool should be of type function call.
        if response.choices[0].message.tool_calls is not None and len(response.choices[0].message.tool_calls) > 0:
            for tool_call in response.choices[0].message.tool_calls:
                if type(tool_call) is ChatCompletionsToolCall:
                    function_args = json.loads(tool_call.function.arguments.replace("'", '"'))
                    print(f"Calling function `{tool_call.function.name}` with arguments {function_args}")
                    callable_func = globals()[tool_call.function.name]
                    function_response = callable_func(**function_args)
                    print(f"Function response = {function_response}")
                    # Provide the tool response to the model, by appending it to the chat history
                    messages.append(ToolMessage(function_response, tool_call_id=tool_call.id))
                    # With the additional tools information on hand, get another response from the model
            response = client.complete(messages=messages, tools=[weather_description, temperature_in_city])

    print(f"Model response = {response.choices[0].message.content}")


def main():
    # [START instrument_inferencing]
    from azure.ai.inference.tracing import AIInferenceInstrumentor

    # Instrument AI Inference API
    AIInferenceInstrumentor().instrument()
    # [END instrument_inferencing]

    try:
        endpoint = os.getenv("AZURE_INFERENCE_SDK_ENDPOINT")
        key = os.getenv("AZURE_INFERENCE_SDK_KEY")
    except KeyError:
        print("Missing environment variable 'AZURE_INFERENCE_SDK_ENDPOINT' or 'AZURE_INFERENCE_SDK_KEY'")
        print("Set them before running this sample.")
        exit()

    chat_completion_with_function_call(key, endpoint)
    # [START uninstrument_inferencing]
    AIInferenceInstrumentor().uninstrument()
    # [END uninstrument_inferencing]


if __name__ == "__main__":
    main()