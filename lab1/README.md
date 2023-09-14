# ML Toolkit

A toolkit for the Course EN.605.649.01 - Intro to Machine Learning.

## Table of Contents

- [Environment Setup](#environment-setup)
- [Execution Instructions](#execution-instructions)
  - [Running Datasets](#running-datasets)
  - [Running Tests](#running-tests)
- [Contributors](#contributors)

## Environment Setup

To set up the Python environment, follow the steps below:

1. Clone the repository:

    ```bash
    git clone https://github.com/TimAgro/JHU-IntroML.git
    cd lab1
    ```

2. Set up a virtual environment:

    ```bash
    python -m venv env
    # On Windows use `env\Scripts\activate`
    source env/bin/activate
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Execution Instructions

### Running Datasets

To execute the code across various datasets and save the results:

```bash
python main.py > ./results/main_log.txt
```

### Running Tests

For unit testing:

```bash
python -m unittest discover -s test
```

## Contributors

### Tim Chen

- 📧 Email: [tchen124@jhu.edu](mailto:tchen124@jhu.edu)
- 🌐 LinkedIn: [Tim Chen](https://www.linkedin.com/in/tim-chen-017b841a9/)
- 🐱‍💻 GitHub: [@TimAgro](https://github.com/TimAgro)
