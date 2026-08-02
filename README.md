# U-Shaped Split Learning Architecture for Privacy-Preserving Predictive Maintenance

## Overview

This project implements a **U-Shaped Split Learning (USL)** framework for distributed predictive maintenance using deep learning. The primary objective is to train a collaborative model across multiple clients while ensuring that **raw sensor data and ground-truth labels never leave the client device**, thereby significantly improving privacy compared to conventional centralized learning and traditional Split Learning.

Unlike standard Split Learning, which divides the neural network into **two parts (client-server)**, the proposed U-Shaped architecture partitions the network into **three models**:

- **Front Model (Client)**
- **Middle Model (Server)**
- **Back Model (Client)**

This architecture enables the client to perform both the initial feature extraction and the final classification, allowing the **loss function and labels to remain entirely on the client**. Only intermediate activations are exchanged with the server, reducing the risk of sensitive information leakage.

The repository also contains an implementation of **Normal Split Learning**, allowing direct comparison between both approaches in terms of communication overhead, model accuracy, computational cost, and privacy.

---

# Key Features

- U-Shaped Split Learning implementation
- Traditional Split Learning implementation
- Three-model neural network architecture
- Privacy-preserving collaborative training
- Multi-client distributed learning
- Client-server communication using sockets
- Performance comparison between USL and Normal Split Learning
- Automatic metrics logging
- Privacy leakage analysis
- Interactive dashboard
- Automatic plot generation

---

# Proposed U-Shaped Architecture

Unlike traditional Split Learning, where the server performs the final classification and computes the loss, the proposed framework splits the neural network into **three independent components**.

```
                Client                         Server                      Client

           ┌────────────┐              ┌────────────┐              ┌────────────┐
Input ───► │ Front Model│ ───────────► │Middle Model│ ───────────► │ Back Model │ ───► Loss
           └────────────┘              └────────────┘              └────────────┘
                 ▲                                                       │
                 └────────────────── Backpropagation ─────────────────────┘
```

### Front Model

The Front Model resides on the client and performs the initial feature extraction from raw sensor data.

Responsibilities:

- Reads raw input data
- Extracts low-level features
- Sends only intermediate activations to the server

Since raw data never leaves the client, data privacy is preserved.

---

### Middle Model

The Middle Model resides on the server.

Responsibilities:

- Receives intermediate activations
- Learns higher-level feature representations
- Returns transformed activations back to the client

The server never has access to:

- Raw sensor data
- Ground-truth labels
- Final predictions

---

### Back Model

The Back Model is placed back on the client.

Responsibilities:

- Performs final classification
- Computes training loss
- Stores labels locally
- Initiates gradient backpropagation

Keeping the classifier on the client prevents the server from learning label information, providing an additional layer of privacy over traditional Split Learning.

---

# Why U-Shaped Split Learning?

Compared to conventional Split Learning:

| Traditional Split Learning | Proposed U-Shaped Split Learning |
|----------------------------|----------------------------------|
| Two-model architecture | Three-model architecture |
| Labels sent to server | Labels remain on client |
| Server computes loss | Client computes loss |
| Higher privacy leakage | Lower privacy leakage |
| Exposes classifier to server | Classifier remains private |

Benefits include:

- Improved privacy
- Reduced label leakage
- Better protection against feature inversion attacks
- Suitable for industrial IoT deployments
- Scalable client-server architecture

---

# Workflow

1. Client loads local dataset.
2. Front Model extracts features.
3. Activations are sent to the server.
4. Server executes the Middle Model.
5. Intermediate activations are returned.
6. Client executes the Back Model.
7. Loss is computed locally.
8. Gradients propagate back through the server.
9. Parameters are updated.
10. Process repeats until convergence.

---

# Performance Evaluation

The repository compares U-Shaped Split Learning against Normal Split Learning using:

- Training Accuracy
- Validation Accuracy
- Communication Cost
- Training Time
- Client Computation
- Server Computation
- Privacy Metrics
- Communication Overhead

---

# Repository Structure

```
new_train/
new_test/

client_split.py
server_split.py

client_normal_split.py
server_normal_split.py

dashboard_app.py
app.py

privacy_analysis.py
auto_compare.py
auto_plot.py

shared_utils.py

requirements.txt
```

---

# Installation

Clone the repository

```bash
git clone https://github.com/<username>/<repository>.git

cd <repository>
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Project

## 1. Run U-Shaped Split Learning

### Start the Server

```bash
python server_split.py
```

### Start Client 1

```bash
python client_split.py
```

Run additional client instances in separate terminals if required.

---

## 2. Run Traditional Split Learning

### Start the Server

```bash
python server_normal_split.py
```

### Start Client

```bash
python client_normal_split.py
```

---

## 3. Launch the Dashboard

```bash
python dashboard_app.py
```

or

```bash
python app.py
```

depending on your configuration.

---

## 4. Compare Both Architectures

```bash
python auto_compare.py
```

---

## 5. Generate Performance Graphs

```bash
python auto_plot.py
```

---

## 6. Run Privacy Analysis

```bash
python privacy_analysis.py
```

---

# Technologies Used

- Python
- PyTorch
- Socket Programming
- Flask
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- HTML
- CSS
- JavaScript

---

# Applications

- Predictive Maintenance
- Industrial IoT
- Smart Manufacturing
- Industry 4.0
- Edge AI
- Privacy-Preserving Deep Learning
- Distributed Machine Learning
- Collaborative Industrial Analytics

---

# Future Enhancements

- Differential Privacy
- Secure Aggregation
- Gradient Compression
- Homomorphic Encryption
- Multi-server Split Learning
- Dynamic Model Partitioning
- Asynchronous Training
- Real-time Edge Deployment

---
