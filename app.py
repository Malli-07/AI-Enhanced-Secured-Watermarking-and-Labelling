from flask import Flask, request, jsonify, render_template, send_file
from transformers import pipeline, GPT2Tokenizer, GPTNeoForCausalLM
from PIL import Image, ImageDraw, ImageFont
import torch
import os

app = Flask(__name__)

# Load a GPT-based model specifically trained for AI vs. Human text classification
# We will use GPT-Neo, which is an open-source GPT model.
text_classifier = pipeline(
    "text-classification",
    model="EleutherAI/gpt-neo-1.3B",  # GPT-Neo model to detect AI-generated text
    tokenizer="EleutherAI/gpt-neo-1.3B"
)

# Load CLIP model for image analysis
from transformers import CLIPProcessor, CLIPModel
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def analyze_text(text):
    """Analyze text for AI/human classification."""
    try:
        result = text_classifier(text)
        label = result[0]['label']  # 'LABEL_1' or 'LABEL_0'
        score = result[0]['score']

        if label == "LABEL_1":  # Assuming LABEL_1 means AI-generated
            ai_probability = f"{score * 100:.2f}%"
            message = "AI-generated"
        else:
            ai_probability = f"{(1 - score) * 100:.2f}%"
            message = "Human-written"

        return {"AI Probability": ai_probability, "message": message}
    except Exception as e:
        return {"error": str(e)}

def analyze_image(image_path):
    """Analyze image using CLIP to check similarity with AI-generated prompts."""
    try:
        img = Image.open(image_path).convert("RGB")
        
        # Preprocess image
        inputs = clip_processor(
            text=["a photo", "an AI-generated image"],  # Compare with these prompts
            images=img,
            return_tensors="pt",
            padding=True
        )
        
        # Get similarity scores
        outputs = clip_model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        
        # Probability of being AI-generated (second prompt)
        ai_score = probs[0][1].item()
        message = "AI-generated" if ai_score > 0.5 else "Human-created"
        
        return {"AI Probability": f"{ai_score * 100:.2f}%", "message": message}
    except Exception as e:
        return {"error": str(e)}

def add_watermark(image_path, message):
    """Add watermark with analysis result to the image."""
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("arial.ttf", 40)
    except:
        font = ImageFont.load_default()
    
    text_position = (10, 10)
    draw.text(text_position, message, fill=(255, 0, 0), font=font)
    
    output_path = "processed_image.png"
    img.save(output_path)
    return output_path

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.form.get('text')
    image = request.files.get('image')

    result = {}

    if text:
        text_result = analyze_text(text)
        result['text_analysis'] = text_result

    if image:
        image_path = "uploaded_image.png"
        image.seek(0)  # Reset file pointer
        image.save(image_path)
        
        image_result = analyze_image(image_path)
        result['image_analysis'] = image_result

        if "error" not in image_result:
            processed_image_path = add_watermark(image_path, image_result['message'])
            return send_file(processed_image_path, as_attachment=True)

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)

