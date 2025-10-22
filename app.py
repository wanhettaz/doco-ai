import gradio as gr
from query import query, load_index

# preload the index so first query isn't slow
print("loading index...")
index = load_index()
print("ready to roll")

def chat(message, history):
    """
    gradio chat interface function
    history lets you do multi-turn convos if you want to extend this
    """
    try:
        response = query(message)
        return str(response)
    except Exception as e:
        return f"error: {str(e)}"

# gradio's chatinterface is dead simple for this use case
demo = gr.ChatInterface(
    fn=chat,
    title="private ai file assistant",
    description="ask questions about your indexed documents",
    examples=[
        "summarise the main concepts",
        "what files mention machine learning?",
        "give me a tldr of the project documentation"
    ],
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",  # accessible on your network
        server_port=7860,
        share=False  # set True if you want a public gradio link
    )