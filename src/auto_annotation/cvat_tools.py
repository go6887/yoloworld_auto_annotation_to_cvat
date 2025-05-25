"""Utilities for converting YOLO detection results to CVAT XML format."""

import xml.etree.ElementTree as ElementTree
from pathlib import Path
from typing import Dict, List

from lxml import etree as ET


def process_image_for_xml(image_id: int, image_data: Dict, detection_results: List[Dict]):
    """Create a <image> element for CVAT XML from detection results."""
    file_name = image_data["file_name"]
    width = image_data["width"]
    height = image_data["height"]

    image_el = ET.Element("image")
    image_el.set("id", str(image_id))
    image_el.set("name", file_name)
    image_el.set("width", str(width))
    image_el.set("height", str(height))

    for obj in detection_results:
        box = obj["box"]
        box_el = ET.SubElement(image_el, "box")
        box_el.set("label", obj["name"])
        box_el.set("occluded", "0")
        box_el.set("source", "manual")
        box_el.set("xtl", str(box["x1"]))
        box_el.set("ytl", str(box["y1"]))
        box_el.set("xbr", str(box["x2"]))
        box_el.set("ybr", str(box["y2"]))
        box_el.set("z_order", "0")
        box_el.set("attributes", "")

    return image_el


def convert_to_cvat_xml(
    images_annotations: List[ET.Element],
    task_info: Dict,
    category_mapping: Dict,
    output_file: Path,
):
    """Convert a collection of <image/> elements into a CVAT XML file."""
    annotations = ET.Element("annotations")
    annotations.set("version", "1.1")

    # Meta information ----------------------------------------------
    meta = ET.SubElement(annotations, "meta")
    task = ET.SubElement(meta, "task")
    ET.SubElement(task, "id").text = str(task_info["id"])
    ET.SubElement(task, "name").text = task_info["name"]
    ET.SubElement(task, "size").text = str(len(images_annotations))
    ET.SubElement(task, "mode").text = "annotation"

    labels_el = ET.SubElement(task, "labels")
    for class_id, cat in category_mapping.items():
        label_el = ET.SubElement(labels_el, "label")
        ET.SubElement(label_el, "name").text = cat["name"]
        ET.SubElement(label_el, "color").text = "000000"
        ET.SubElement(label_el, "attributes")

    for image_el in images_annotations:
        annotations.append(image_el)

    # Pretty-printing helper
    def _indent(elem, level: int = 0):
        i = "\n" + level * "  "
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + "  "
            for child in elem:
                _indent(child, level + 1)
            if not child.tail or not child.tail.strip():
                child.tail = i
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i

    _indent(annotations)

    tree = ET.ElementTree(annotations)
    tree.write(str(output_file), pretty_print=True, xml_declaration=True, encoding="utf-8")


def sort_xml_images_by_name(input_xml_file: Path, output_xml_file: Path):
    """Sort <image> elements by their 'name' attribute in-place."""
    tree = ElementTree.parse(str(input_xml_file))
    root = tree.getroot()

    images = root.findall("image")
    sorted_images = sorted(images, key=lambda x: x.get("name"))

    # Remove existing <image> tags
    for image in images:
        root.remove(image)

    # Append sorted images with refreshed id attributes
    for idx, image in enumerate(sorted_images, start=0):
        image.set("id", str(idx))
        root.append(image)

    tree.write(str(output_xml_file), encoding="utf-8", xml_declaration=True)