from groq import Groq
import pandas as pd
import os
# GROQ_API_KEY = "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"
client = Groq(api_key = "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")
df = pd.read_csv('generated_prompts.csv')
columns = ['email', 'subject', 'body', 'email_class']
data = pd.DataFrame(columns=columns)
for index, row in df.iterrows():
    chat_completion = client.chat.completions.create(
        #
        # Required parameters
        #
        messages=[
            # Set an optional system message. This sets the behavior of the
            # assistant and can be used to provide specific instructions for
            # how it should behave throughout the conversation.
            {
                "role": "system",
                "content": "you are a helpful assistant."
            },
            # Set a user message for the assistant to respond to.
            {
                "role": "user",
                "content": df['prompt'][index],
            }
        ],

        # The language model which will generate the completion.
        model="llama-3.1-70b-versatile",

        #
        # Optional parameters
        #

        # Controls randomness: lowering results in less random completions.
        # As the temperature approaches zero, the model will become deterministic
        # and repetitive.
        temperature=0.5,

        # The maximum number of tokens to generate. Requests can use up to
        # 32,768 tokens shared between prompt and completion.

        # Controls diversity via nucleus sampling: 0.5 means half of all
        # likelihood-weighted options are considered.
        top_p=1,

        # A stop sequence is a predefined or user-specified text string that
        # signals an AI to stop generating content, ensuring its responses
        # remain focused and concise. Examples include punctuation marks and
        # markers like "[end]".
        stop=None,

        # If set, partial message deltas will be sent.
        stream=False,
    )

    # Print the completion returned by the LLM.
    # print(chat_completion.choices[0].message.content)
    # print("-----------------------------------------------------")
    model_output = chat_completion.choices[0].message.content
    intro_text = 'Here are 30 emails for the "{}" class in the "{}" category:\n'
    model_output = model_output.replace(intro_text, '').strip()
    # Split the output into individual email entries
    email_entries = model_output.split('\n')
    # Loop through each generated email and append it to the DataFrame
    for email in email_entries:
        email_parts = email.split('","')
        email_parts = [part.strip('"') for part in email_parts]  # Remove leading/trailing quotes
        print(email_parts)
        if len(email_parts) == 4:
            data = data.append(pd.Series(email_parts, index=columns), ignore_index=True)
data.to_csv('generated_emails.csv', index=False)
print(data.head())
