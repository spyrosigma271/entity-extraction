# Entity Extraction

This project provides a user-friendly tool for extracting entities from documents based on a schema file (e.g., CSV or JSON). The schema specifies field names and their descriptions, enabling customized and scalable entity extraction workflows.

---

## Features
- Upload document and schema files to extract entities.
- Supports structured and unstructured documents.
- Scalable and user-friendly interface built using Streamlit.
- Modular design for extensibility and customization.

---

## Installation

### Step 1: Clone the Repository
```bash
https://github.com/spyrosigma271/entity-extraction.git
```

### Step 2: Create a Virtual Environment

It is recommended to use a virtual environment to manage dependencies.

#### On Linux/Mac:
```bash
python3 -m venv venv
source venv/bin/activate
```

#### On Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies

Install the required Python libraries:
```bash
pip install -r requirements.txt
```

---

## Usage

### Run the Application
Start the Streamlit application:
```bash
streamlit run app.py
```

### Upload Files
1. **Schema File**: Upload a schema file (CSV or JSON) containing field names and descriptions.
2. **Document File**: Upload the document from which you want to extract entities.

### Extract Entities
The application will process the uploaded files and display the extracted entities in an interactive interface. You can also download the results as a file.

---

## Schema Format
The schema file should be in CSV or JSON format and must include the following fields:
- **Field Name**: The name of the entity to be extracted.
- **Description**: A description of the entity to guide the extraction process.

### Example Schema (CSV):
```csv
Field Name,Description
Name,The full name of a person
Email,Email address of the individual
Date,Date in YYYY-MM-DD format
```

---

## Dependencies
- Python 3.7+
- Streamlit
- Additional dependencies listed in `requirements.txt`

---

## Contributing
1. Fork the repository.
2. Create a new branch for your feature (`git checkout -b feature-name`).
3. Commit your changes (`git commit -m "Add feature"`).
4. Push to the branch (`git push origin feature-name`).
5. Open a pull request.

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.

---

## Support
For issues or questions, please open an issue in the repository or contact the maintainer.

