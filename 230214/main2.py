from torch.utils.data import Dataset
from xml.etree.ElementTree import parse

def box_xyxy(box_metas):
    list_box = []
    for image_meta in image_metas :
        box_metas = img_meta.findall("box")
        for box_meta in box_metas:
            box_label = box_meta.attrib['label']
            box [int(float(box_meta.attrib['xt1'])),
                 int(float(box_meta.attrib['yt1'])),
                 int(float(box_meta.attrib['xbr'])),
                 int(float(box_meta.attrib['ybr']))]
            list_box.append(box)
        return list_box

class CustomDataset(Dataset):

    def __init(self, dataset_path, transform=None):
        self.dataset_path = dataset_path
        self.xml_path = xml_path

    def __getitem__(self, index):
        image_path = self.dataset_path[index]
        xml_path = self.xml_path[index]
        print(xml_path)
        tree = parse(xml_path)
        root = tree.getroot()
        image_metas = root.findall('image')

        box_xyxy(box_metas)

        # aug

        print(image_metas)

    def __len(self):
        return len(self.dataset_path)

image_path = ["./01.png"]
xml_path = ["./annotations.xml"]

    