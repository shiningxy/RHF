from PIL import Image
import os
import xml.etree.ElementTree as ET
import warnings


def check_annotations(root):
    xml_path = os.path.join(root, "Annotations")
    xmls = os.listdir(xml_path)
    with_mask = []
    poor_mask = []
    none_mask = []
    all_annotations = 0
    for xml in xmls:
        doc = ET.parse(os.path.join(xml_path, xml))
        root = doc.getroot()
        for child in root:
            if child.tag == "object":
                sub = child.find('name')  # 找到object->name标签
                all_annotations = all_annotations + 1
                if sub.text == "with_mask":
                    with_mask.append(xml.split('.')[0])
                if sub.text == "poor_mask":
                    poor_mask.append(xml.split('.')[0])
                if sub.text == "none_mask":
                    none_mask.append(xml.split('.')[0])
    print("num of all_annotations: ", str(all_annotations))
    print("num of with_mask: ", len(with_mask))
    print("num of poor_mask: ", len(poor_mask))
    print("num of none_mask: ", len(none_mask))
    return with_mask, poor_mask, none_mask


# 检查数据集中图片和标签个数与图片格式是否都为jpg 如果是png则pngs2jpgs
def check_datasets(root):
    xml_path = os.path.join(root, "Annotations")
    jpg_path = os.path.join(root, "JPEGImages")
    xmls = os.listdir(xml_path)
    jpgs = os.listdir(jpg_path)
    # 替换列表中的数据仅为图片和标签无后缀的文件名
    for i in range(0, len(xmls)):
        xmls[i] = xmls[i].split(".")[0]
    for i in range(0, len(jpgs)):
        jpgs[i] = jpgs[i].split(".")[0]
    # 只留取公共的jpg和xml
    common = [x for x in xmls if x in jpgs]
    # addition_jpgs : 没有相应标签的图片列表
    addition_jpgs = [y for y in jpgs if y not in xmls]
    for i in range(0, len(addition_jpgs)):
        addition_jpgs[i] = addition_jpgs[i] + ".jpg"
    # 删除多余文件
    for i in addition_jpgs:
        os.remove(os.path.join(os.getcwd(), os.path.join(jpg_path, i)))
    print("已删除%s个多余的标签文件" % len(addition_jpgs))
    # addition_xmls : 没有相应图片的标签列表
    addition_xmls = [y for y in xmls if y not in jpgs]
    for i in range(0, len(addition_xmls)):
        addition_xmls[i] = addition_xmls[i] + ".xml"
    # 删除多余文件
    for i in addition_xmls:
        os.remove(os.path.join(os.getcwd(), os.path.join(xml_path, i)))
    print("已删除%s个多余的图片文件" % len(addition_xmls))
    print("标签与图片数量相等")
    print("addition_jpgs : ", end='')
    print(addition_jpgs)
    print("addition_xmls : ", end='')
    print(addition_xmls)
    print("共有%s个文件相同" % (len(common)))
    print("共有%s个jpg文件不同" % (len(addition_jpgs)))
    print("共有%s个xml文件不同" % (len(addition_xmls)))


def check_format(root):
    xml_path = os.path.join(root, "Annotations")
    img_path = os.path.join(root, "JPEGImages")
    xmls = os.listdir(xml_path)
    imgs = os.listdir(img_path)
    right_format_list = []
    for img in imgs:
        path = os.path.join(img_path, img)
        try:
            image = Image.open(path)
        except:
            os.remove(os.path.join(img_path, img))
            os.remove(os.path.join(xml_path, img.split('.')[0] + ".xml"))
            print("remove broken image successful!")
            continue
        if image.format not in ["JPEG", "JPG", "jpg", "jpeg"] and image.format in ['png', "PNG"]:
            print("still have png")
        elif image.format in ["JPEG", "JPG", "jpeg", "jpg"]:
            right_format_list.append(img)
        else:
            os.remove(os.path.join(img_path, img))
            os.remove(os.path.join(xml_path, img.split('.')[0] + ".xml"))
            print("remove image.format not in png and jpg successful")

    return right_format_list


if __name__ == '__main__':
    warnings.filterwarnings("error", category=UserWarning)
    check_datasets("F:/face-mask-detect/MaskDatasets-Light")
    check_annotations("F:/face-mask-detect/MaskDatasets-Light")
    # print("正确图片共有 : %s 张 " % str(len(check_format("F:/face-mask-detect/MaskPascalVOC"))))
