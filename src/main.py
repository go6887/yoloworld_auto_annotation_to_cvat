import glob
import os
import xml.etree.ElementTree as ElementTree
import zipfile
from pathlib import Path

from lxml import etree as ET
from ultralytics import YOLO

# 検知したい対象を指定
# ここでは道路標識を検知するように設定
DETECTION_TARGET = ["road sign"]

# yoloのモデルを読み込み
model = YOLO("yolov8s-world.pt")
model.set_classes(DETECTION_TARGET)

def sort_xml_images_by_name(input_xml_file, output_xml_file):
    """
    指定されたXMLファイルを読み込み、<image>要素を'name'属性でソートし、
    'id'属性を振り直して新しいXMLファイルに保存します。

    :param input_xml_file: 入力XMLファイルのパス
    :param output_xml_file: 出力XMLファイルのパス
    """
    # XMLファイルの読み込み
    tree = ElementTree.parse(input_xml_file)
    root = tree.getroot()

    # <image>要素の取得
    images = root.findall('image')

    # 'name'属性でソート
    sorted_images = sorted(images, key=lambda x: x.get('name'))

    # 既存の<image>要素を削除
    for image in images:
        root.remove(image)

    # ソートされた<image>要素を追加し、idを振り直す
    for idx, image in enumerate(sorted_images, start=0):
        image.set('id', str(idx))
        root.append(image)

    # XMLファイルの保存
    tree.write(output_xml_file, encoding='utf-8', xml_declaration=True)

def zip_directory(output_dir, zip_filename):
    """
    指定したディレクトリをZIPファイルに圧縮する関数。

    :param output_dir: 圧縮対象のディレクトリのパス
    :param zip_filename: 作成するZIPファイルの名前
    """
    output_path = Path(output_dir)
    
    # ディレクトリの存在確認
    if not output_path.exists() or not output_path.is_dir():
        raise FileNotFoundError(f"指定されたディレクトリが存在しません: {output_dir}")

    # ZIPファイルの作成
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                file_path = Path(root) / file
                # ZIP内のファイルパスを設定（親ディレクトリからの相対パス）
                zipf.write(file_path, file_path.relative_to(output_path.parent))
                print(f"追加: {file_path}")

    print(f"ディレクトリ '{output_dir}' を '{zip_filename}' に圧縮しました。")

# CVAT XML生成に必要な関数
def process_image_for_xml(image_id, image_data, detection_results):
    """
    検出対象が含まれている画像のアノテーションデータをCVATの<image>要素として生成する関数。
    :param image_id: 画像のID
    :param image_data: 画像情報とオブジェクトリスト
    :param detection_results: 検出結果のリスト
    :return: <image>要素
    """
    file_name = image_data['file_name']
    width = image_data['width']
    height = image_data['height']
    objects = detection_results  # 検出されたオブジェクトのみ
    # <image>要素の作成
    image_el = ET.Element("image")
    image_el.set("id", str(image_id))
    image_el.set("name", file_name)
    image_el.set("width", str(width))
    image_el.set("height", str(height))
    # 各オブジェクトの追加
    for obj in objects:
        box = obj['box']
        box_el = ET.SubElement(image_el, "box")
        box_el.set("label", obj['name'])
        box_el.set("occluded", "0")
        box_el.set("source", "manual")
        box_el.set("xtl", str(box['x1']))
        box_el.set("ytl", str(box['y1']))
        box_el.set("xbr", str(box['x2']))
        box_el.set("ybr", str(box['y2']))
        box_el.set("z_order", "0")
        box_el.set("attributes", "")  # 属性があればここに追加
    return image_el

def convert_to_cvat_xml(images_annotations, task_info, category_mapping, output_file):
    """
    複数の画像データをCVAT XMLフォーマットに変換する関数。
    :param images_annotations: 画像ごとの<image>要素のリスト
    :param task_info: タスク情報の辞書（id, name）
    :param category_mapping: カテゴリのマッピング辞書（クラスID -> カテゴリ名）
    :param output_file: 出力するXMLファイルのパス
    """
    # ルート要素の作成
    annotations = ET.Element("annotations")
    annotations.set("version", "1.1")
    # メタ情報の作成
    meta = ET.SubElement(annotations, "meta")
    task = ET.SubElement(meta, "task")
    # タスク情報の追加
    ET.SubElement(task, "id").text = str(task_info['id'])
    ET.SubElement(task, "name").text = task_info['name']
    ET.SubElement(task, "size").text = str(len(images_annotations))  # 画像数
    ET.SubElement(task, "mode").text = "annotation"
    # ラベルの追加
    labels_el = ET.SubElement(task, "labels")
    for class_id, cat in category_mapping.items():
        label_el = ET.SubElement(labels_el, "label")
        ET.SubElement(label_el, "name").text = cat['name']
        ET.SubElement(label_el, "color").text = "000000"  # デフォルト色
        ET.SubElement(label_el, "attributes")  # 属性があればここに追加
    # 生成された<image>要素をルートに追加
    for image_el in images_annotations:
        annotations.append(image_el)
    # インデントを整えるための関数
    def indent(elem, level=0):
        i = "\n" + level * "  "
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + "  "
            for child in elem:
                indent(child, level + 1)
            if not child.tail or not child.tail.strip():
                child.tail = i
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i
    indent(annotations)
    # XMLツリーの作成と保存
    tree = ET.ElementTree(annotations)
    tree.write(output_file, pretty_print=True, xml_declaration=True, encoding="utf-8")

def execute_yolo_inference(image_path):  
    results = model.predict(
        source=image_path
    )
    # debug用で結果を表示したい場合はコメントアウトを外す
    # results[0].show()
    return results[0].summary()

def main():
    # 画像が格納されているディレクトリを指定
    input_directory = "20241210/images"
    output_dir = "20241210"
    os.makedirs(output_dir, exist_ok=True)

    # タスク情報
    task_info = {
        "id": 1,
        "name": "CVAT ANNOTATION",
    }

    # カテゴリマッピングの定義（クラスID -> カテゴリ名）
    category_mapping = {
        0: {"name": "road sign"},
        # 他のクラスがあればここへ追加
    }

    images_annotations = []
    image_id = 0

    # jpg画像をすべて取得（png等、他形式にも対応したい場合は拡張子を変更・追加）
    for image_path in glob.glob(os.path.join(input_directory, "*.jpg")):
        # YOLO 推論
        result = execute_yolo_inference(image_path)
        print(f"{image_path} の推論結果:", result)

        # ---- 実際に画像サイズを取得したい場合 (PIL を利用) のサンプル
        # from PIL import Image
        # with Image.open(image_path) as img:
        #     width, height = img.size
        # ただし以下では固定値480x640としています

        width, height = 480, 640

        image_info = {
            "file_name": os.path.basename(image_path),
            "width": width,
            "height": height,
        }

        annotation = process_image_for_xml(image_id, image_info, result)
        images_annotations.append(annotation)
        image_id += 1

    # CVAT形式XML作成用の出力ファイルパス
    output_file = os.path.join(output_dir, "annotations.xml")

    # images_annotations, task_info, category_mapping をもとにCVAT形式のXMLを作成
    convert_to_cvat_xml(images_annotations, task_info, category_mapping, output_file)

    # 画像ファイル名でソート
    sort_xml_images_by_name(output_file, output_file)

    # zip化
    try:
        zip_directory(output_dir, "sample.zip")
        print("処理が完了しました。")
    except Exception as e:
        print(f"エラーが発生しました: {e}")

if __name__ == "__main__":
    main()