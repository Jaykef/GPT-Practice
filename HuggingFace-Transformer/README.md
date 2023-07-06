# 🤗 Transformers
State-of-the-art Machine Learning for PyTorch, TensorFlow, and JAX.

🤗 Transformers provides APIs and tools to easily download and train state-of-the-art pretrained models. Using pretrained models can reduce your compute costs, carbon footprint, and save you the time and resources required to train a model from scratch. These models support common tasks in different modalities, such as:
<ul>
  <li>
📝 Natural Language Processing: text classification, named entity recognition, question answering, language modeling, summarization, translation, multiple choice, and text generation.</li>
  <li>🖼️ Computer Vision: image classification, object detection, and segmentation.</li>
  <li>🗣️ Audio: automatic speech recognition and audio classification.</li>
  <li>🐙 Multimodal: table question answering, optical character recognition, information extraction from scanned documents, video classification, and visual question answering.</li>
</ul>

🤗 Transformers support framework interoperability between PyTorch, TensorFlow, and JAX. This provides the flexibility to use a different framework at each stage of a model’s life; train a model in three lines of code in one framework, and load it for inference in another. Models can also be exported to a format like ONNX and TorchScript for deployment in production environments.Ω

<a href="https://huggingface.co/docs/transformers/installation">Installation </a>

Start by creating a virtual environment in your project directory:
```
python -m venv .env
```
Activate the virtual environment. On Linux and MacOs:
```
source .env/bin/activate
```
Activate Virtual environment on Windows

```
.env/Scripts/activate
```
Now you’re ready to install 🤗 Transformers with the following command:

```
pip install transformers
```

