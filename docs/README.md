# MLC-LLM Documentation

The documentation was built upon [Sphinx](https://www.sphinx-doc.org/en/master/).

## Dependencies

Run the following command in this directory to install dependencies first:

```bash
pip3 install -r requirements.txt
```

## Build the Documentation

Then you can build the documentation by running:

```bash
make html
```

## View the Documentation

Run the following command to start a simple HTTP server:

```bash
cd _build/html
python3 -m http.server
```

Then you can view the documentation in your browser at `http://localhost:8000` (the port can be customized by appending ` -p PORT_NUMBER` in the python command above).
