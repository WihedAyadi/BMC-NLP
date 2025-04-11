# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 21:53:59 2025

@author: HP
"""


from sentence_transformers import SentenceTransformer
import numpy as np
from collections import Counter
import pandas as pd
import faiss
import spacy
from spacy.util import minibatch
import random
from spacy.training import Example
import pdfplumber
import os


def extract_paragraphs_from_pdf(pdf_path):
    paragraphs = []
    
    # Open the PDF
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            # Extract text (pdfplumber preserves line breaks better than PyPDF2)
            text = page.extract_text()
            
            # Split into paragraphs (assuming paragraphs are separated by "\n\n")
            page_paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            
            # Add to the main list
            paragraphs.extend(page_paragraphs)
    
    return paragraphs
def get_pdf_files(directory):
    """
    Returns a list of full paths to all PDF files in the specified directory.

    Args:
        directory (str): Path to the directory to search for PDF files

    Returns:
        list: List of full file paths to PDF files in the format:
              ["C:\\path\\to\\file1.pdf", "C:\\path\\to\\file2.pdf"]
    """
    pdf_files = []

    # Verify directory exists
    if not os.path.isdir(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        return pdf_files

    # Get all files in directory
    for filename in os.listdir(directory):
        if filename.lower().endswith('.pdf'):
            # Get full path and add to list
            full_path = os.path.join(directory, filename)
            pdf_files.append(full_path)

    return pdf_files
# Prepare training examples
train_dataset = {
    "Key partners": [
        "Who are our key suppliers?",
        "Which Key Resources are we acquiring from partners?",
        " Key Partnerships refer to the network of external organizations, suppliers, or collaborators that help a business achieve its objectives and execute its business model. These partnerships provide essential resources, activities, or capabilities that the company might not have in-house, enabling it to focus on its core operations.",
        "OVO Energy: Collaborating with SMEs for large-scale energy management solutions.",
"Power Ledger: Facilitating blockchain-based decentralized trading models for SMEs.",
"Western Power: Providing grid management for SMEs generating renewable energy.",
"Synergy: Assisting SMEs with energy supply, management, and optimization.,"
    ],
    "Key Activities": [
        "What Key Activities do our Value Propositions require?",
        "Which Key Activities are most important for our business?",
        "Key Activities are the essential actions a company must take to ensure its business model functions effectively. These activities are crucial for creating and delivering the Value Proposition, reaching Customer Segments, maintaining Customer Relationships, and generating Revenue Streams",
        "Deployment of advanced energy management systems.",
"Investment in solar energy solutions and battery storage to optimize energy consumption.",
"Integration of blockchain-based energy trading systems to allow direct trading with other prosumers."
    ],
    "Value Propositions": [
        "What value do we deliver to the customer?",
        "Which customer needs are we satisfying?",
        " Value Proposition refers to the set of products and services that deliver unique value to a specific customer segment. It defines how a business solves customer problems or fulfills their needs, differentiating it from competitors. Value can be quantitative, such as cost efficiency or speed of service, or qualitative, such as superior design or an enhanced customer experience. These factors determine how a business creates and delivers value to its target customer segments",
        "Innovative business models that allow SMEs to reduce their energy costs through renewable energy generation and trading.",
"Access to new energy models such as Electricity as a Service (EaaS) and blockchain-powered energy trading.",
"Flexibility and autonomy in managing energy costs, while contributing to a cleaner energy grid.",
    ],
    "Customer Relationships": [
        "What type of relationship does each of our Customer Segments expect?",
        "How are our customer relationships integrated with the business model?",
        "Customer Relationships refer to the types of interactions a company builds with its Customer Segments to attract, retain, and grow its customer base. These relationships can vary based on business goals and customer needs, including Personal assistance, Self-service, Automated services, Communities, Co-creation",
        "Strategic partnerships with energy suppliers offering business solutions.",
"Customer service for energy optimization and trading advice.",
    ],
    "Customer Segments": [
        "Small and Medium Enterprises (SMEs) interested in renewable energy and energy cost reduction.",
        "Businesses participating in blockchain-based energy systems and adopting decentralized models."
    ],
    "Revenue Stream" : ["Cost savings from generating and using self-consumed energy.",
"Revenue from selling surplus energy through decentralized trading models.",
"Energy trading commissions in blockchain-based systems",
"Subscription fees for using blockchain-based platforms",
"P2P market earnings from energy sharing",
"Turn-key solar leasing and PPAs"
"For what value are our customers really willing to pay?",
        "How much does each Revenue Stream contribute to overall revenues?",
        "Revenue Streams represent the cash flow a company earns from each Customer Segment in exchange for delivering a Value Proposition",],
    "Key Resources": [
        "What Key Resources do our Value Propositions require?",
        "Which Key Resources are most expensive?",
        "Key Resources are the essential assets a business needs to operate successfully and deliver its Value Proposition. These resources enable the company to reach its Customer Segments, build and maintain relationships, and generate revenue. They can be categorized as physical, financial, intellectual, or human. Key resources may be owned, leased, or acquired through partnerships, depending on the business model",
        "Blockchain technology for energy transactions",
"Smart metering systems",
"Remote control systems",
"Energy management platforms (e.g., Power Ledger)",
"Advanced energy storage and management systems."],
    "Cost Structure": [
        "What are the most important costs in our business model?",
        "Which Key Activities are most expensive?",
        "Cost Structure represents the total expenses a company incurs to operate its business model. These costs are necessary to deliver the Value Proposition, maintain Customer Relationships, reach Customer Segments, and manage Key Activities, Key Resources, and Key Partnerships",
        "Investment in renewable energy infrastructure (e.g., PV systems, batteries).",
"Operational and maintenance costs associated with energy systems and blockchain integration.",
"Fixed costs for system development and customer management",
"Operational costs for software development and updates",
    ],
    
    "Channels": [" Channels represent the means by which a company communicates with and delivers its Value Proposition to its Customer Segments. These include communication, distribution, and sales channels, forming the company’s interface with customers. Channels can be direct (e.g., company-owned stores, websites, or sales teams), indirect (e.g., partner retailers, third-party distributors), owned (fully controlled by the company), partner-based (leveraging external organizations for wider reach)",
                 "Blockchain platforms for decentralized energy trading.",
"Direct relationships with energy providers offering tailored solutions for businesses.",
"Peer-to-peer digital platforms",
"Online self-service energy platforms",
"Community networks and workshops"
                 ]
}
#initialising the nlp
nlp = spacy.blank("en")
#formatting the data
data = {
    "text": [],
    "label": []
}

for label, texts in train_dataset.items():
    for text in texts:
        # Clean text by stripping whitespace and skipping empty strings
        cleaned_text = text.strip()
        if cleaned_text:  # Only add non-empty texts
            data["text"].append(cleaned_text)
            data["label"].append(label)

# Optional: Convert to pandas DataFrame for easier handling
df = pd.DataFrame(data)

# Verify the output
print(f"Total samples: {len(df)}")
print(df.head())

pd.DataFrame(data).to_parquet("labeled_paragraphs.parquet")

model = SentenceTransformer("all-MiniLM-L6-v2")
df = pd.read_parquet("labeled_paragraphs.parquet")

# Generate embeddings
embeddings = model.encode(df["text"].tolist())
df["embedding"] = embeddings.tolist()

# Save with embeddings
df.to_parquet("labeled_paragraphs_with_embeddings.parquet")

class RagAugmenter:
    def __init__(self, knowledge_base_path):
        """Initialize with a pre-processed knowledge base"""
        self.df = pd.read_parquet(knowledge_base_path)
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Build FAISS index
        embeddings = np.array(self.df["embedding"].tolist()).astype("float32")
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)
    
    def retrieve_similar(self, text, top_k=3):
        """Retrieve top_k most similar paragraphs"""
        query_embedding = self.model.encode([text])
        distances, indices = self.index.search(query_embedding, top_k)
        return self.df.iloc[indices[0]].to_dict("records")
    
    def majority_vote(self, retrieved_items):
        """Determine the most common label among retrieved items"""
        labels = [item["label"] for item in retrieved_items]
        return Counter(labels).most_common(1)[0][0]  # Returns (label, count)
# 1. Extract paragraphs (original code)
paragraphs = extract_paragraphs_from_pdf("/content/Parra-Domínguez-2023-The Prosumer_ A Systemati (1).pdf")

# 2. Initialize RAG augmenter
rag = RagAugmenter("/content/labeled_paragraphs_with_embeddings.parquet")
# Add text classifier to the pipeline
if "textcat" not in nlp.pipe_names:
    textcat = nlp.add_pipe("textcat", last=True)

    # Add all labels to the text classifier
    for label in train_dataset.keys():
        textcat.add_label(label)
else:
    textcat = nlp.get_pipe("textcat")

examples = []
for category, texts in train_dataset.items():
    for text in texts:
        # Create the annotation dictionary
        cats = {label: False for label in train_dataset.keys()}
        cats[category] = True  # Set the true category

        # Create an Example object
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, {"cats": cats})
        examples.append(example)

# 2. Training setup
random.seed(1)
spacy.util.fix_random_seed(1)
optimizer = nlp.begin_training()

# 3. Training loop
epochs = 50
batch_size = 2
losses = {}

print("Beginning training...")
for epoch in range(epochs):
    random.shuffle(examples)
    batches = minibatch(examples, size=batch_size)

    for batch in batches:
        nlp.update(batch, sgd=optimizer, losses=losses)

    print(f"Epoch {epoch + 1}, Loss: {losses['textcat']:.3f}")
# 3. Process each paragraph
results = []
for para in paragraphs:
    # Classify with spaCy first
    doc = nlp(para)
    max_key, max_conf = max(doc.cats.items(), key=lambda x: x[1]) if doc.cats else (None, 0)
    
    # RAG augmentation for low-confidence cases
    if max_conf < 0.5:  # Higher threshold for RAG intervention
        rag_results = rag.retrieve_similar(para, top_k=3)
        max_key = rag.majority_vote(rag_results)  # Get consensus from retrievals
    
    results.append({
        "Paragraph": para,
        "Category": max_key,
        "Confidence": max_conf,
        "RAG_Augmented": max_conf < 0.5  # Flag for RAG-assisted classifications
    })

# 4. Save to Excel
pd.DataFrame(results).to_excel("rag_optimized_results.xlsx", index=False)
