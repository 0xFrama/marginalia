from docling.document_converter import DocumentConverter

source = "attention.pdf"
converter = DocumentConverter()
doc = converter.convert(source).document

for item, level in doc.iterate_items():
    if hasattr(item, "prov") and item.prov:
        print(item.label, item.prov)
        break
    print(type(item).__name__, level, item.label if hasattr(item, "label") else "-")


# print(doc.export_to_markdown())
