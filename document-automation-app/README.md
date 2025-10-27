# Document Automation Application

This document automation application leverages LangChain to streamline the processing of documents. It is designed to extract relevant information and apply transformations to input documents efficiently.

## Project Structure

```
document-automation-app
├── src
│   ├── main.py                # Entry point of the application
│   ├── automation
│   │   └── processor.py       # Contains DocumentProcessor class for document handling
│   ├── langchain_utils
│   │   └── chain.py           # Contains LangChain class for processing chains
│   └── config
│       └── settings.py        # Configuration settings for the application
├── requirements.txt           # Project dependencies
├── README.md                  # Project documentation
└── .gitignore                 # Files and directories to ignore by Git
```

## Setup Instructions

1. **Clone the repository:**
   ```
   git clone <repository-url>
   cd document-automation-app
   ```

2. **Create a virtual environment:**
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```

## Usage

To run the application, execute the following command:

```
python src/main.py
```

## Functionality Overview

- **Document Processing:** The application reads input documents, processes them using the `DocumentProcessor` class, and extracts relevant data.
- **LangChain Integration:** The `LangChain` class facilitates the creation and execution of processing chains, allowing for modular and reusable document processing workflows.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.




### Agent Pipeline

```
Input CSV
    ↓
┌─────────────────────┐
│ Agent 1: Data Loader│  → Load CSV data
└─────────────────────┘
    ↓
┌─────────────────────┐
│ Agent 2: Calculator │  → Calculate BSM prices & Greeks
└─────────────────────┘
    ↓
┌─────────────────────┐
│ Agent 3: Tester     │  → Run validation tests
└─────────────────────┘
    ↓
┌─────────────────────┐
│ Agent 4: Summary    │  → Generate text summary
└─────────────────────┘
    ↓
┌─────────────────────┐
│ Agent 5: Charts     │  → Create visualizations
└─────────────────────┘
    ↓
┌─────────────────────┐
│ Agent 6: Assembler  │  → Create final HTML report
└─────────────────────┘
    ↓
Final Report (HTML)
```