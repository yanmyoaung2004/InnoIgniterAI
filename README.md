# InnoIgniterAI

AI-driven cybersecurity platform for threat detection, security analysis, and intelligent incident response.

---

## Overview

InnoIgniterAI is an AI-powered cybersecurity platform designed to identify, analyze, and explain digital threats through a combination of Large Language Models, multi-agent systems, and security monitoring technologies.

The platform integrates threat detection, knowledge-based assistance, automated security analysis, and security event monitoring into a unified system.

It aims to bridge the gap between advanced cybersecurity tools and everyday users by converting complex security information into understandable and actionable insights.

---

# Core Capabilities

## Threat Detection and Analysis

InnoIgniterAI provides automated analysis for:

- Malicious URLs and phishing attempts
- Suspicious files
- Phishing emails
- Security-related indicators

The system analyzes potential threats and provides explanations with recommended actions.

---

## AI Cybersecurity Assistant

A conversational AI assistant powered by:

- Large Language Models (LLMs)
- Retrieval-Augmented Generation (RAG)
- Context-aware security knowledge retrieval

Capabilities include:

- Cybersecurity question answering
- Security concept explanation
- Threat investigation assistance
- User-friendly security guidance

---

## Multi-Agent AI Architecture

The system uses a modular agent-based architecture where specialized AI agents handle different cybersecurity tasks.

Architecture components:

- Task planning agent
- Threat analysis agent
- Knowledge retrieval agent
- Security monitoring agent
- Response generation agent

Agent coordination is implemented using:

- LangGraph
- MCP-based communication architecture

This allows independent agent development, scaling, and future expansion.

---

## Security Monitoring and SIEM Integration

The application integrates with:

- Wazuh SIEM

Features include:

- Security log collection
- Event analysis
- Threat correlation
- Anomaly detection
- Security alert generation

The SIEM functionality is available in the application version.

---

## Blockchain-Based Evidence Ledger

Security events can be stored using a blockchain-secured evidence ledger.

Purpose:

- Maintain event integrity
- Prevent unauthorized modification
- Provide trustworthy security records

---

## Multilingual Interface

Supports:

- English
- Myanmar

Includes:

- Multilingual cybersecurity assistance
- Voice-based interaction support

---

# System Architecture

```

User Interface
|
|
Frontend Application
|
|
FastAPI Backend
|
|
MCP Controller
|
|
Multi-Agent AI System
|
+----------------+
|                |
|                |
RAG System       Security Agents
|                |
|                |
Vector Database    Wazuh SIEM

```
  |
```

Blockchain Evidence Ledger

```

---

# Technology Stack

| Layer | Technology |
|---|---|
| Backend | Python, FastAPI |
| AI Framework | LangChain, LangGraph |
| AI Architecture | Multi-Agent System, MCP |
| Retrieval System | RAG |
| Security Monitoring | Wazuh SIEM |
| Storage | Vector Database, Structured Logs |
| Frontend | React |
| Communication | REST API |


---

# Frontend Repository

Frontend is maintained separately:

Repository:

https://github.com/yanmyoaung2004/InnoIgniterAI_UI

The frontend communicates with backend APIs to provide real-time cybersecurity analysis and visualization.

---

# Installation

## Clone Repository

```bash
git clone https://github.com/yanmyoaung2004/InnoIgniterAI.git

cd InnoIgniterAI
````

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run Application

```bash
uvicorn main:app --reload
```

The API server will start locally.

---

# Use Cases

## Personal Security

* Detect suspicious links before opening
* Analyze suspicious emails
* Understand security risks

## Enterprise Security

* Monitor security events
* Analyze logs
* Assist security operations

## Security Education

* Explain cybersecurity concepts
* Provide interactive security guidance

---

# Achievements

* Top 10 Finalist — AI Competition
* People's Choice Award Winner

---

# License

Copyright © 2025 Yan Myo Aung.
All rights reserved.

