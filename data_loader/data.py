class Data:
    def __init__(self):
        self.doc_set = {}
        self.qry_set = {}
        self.rel_set = {}
    def readData(self):
        with open('../cisi/CISI.ALL') as f:
            lines = ""
            for l in f.readlines():
                lines += "\n" + l.strip() if l.startswith(".") else " " + l.strip()
            lines = lines.lstrip("\n").split("\n")
        #
        # # print n lines
        # n = 5
        # for l in lines[:n]:
        #     print(l)
        # doc_set = {}
        doc_id = ""
        doc_text = ""
        for l in lines:
            if l.startswith(".I"):
                doc_id = l.split(" ")[1].strip()
            elif l.startswith(".X"):
                self.doc_set[doc_id] = doc_text.lstrip(" ")
                doc_id = ""
                doc_text = ""
            else:
                doc_text += l.strip()[3:] + " " # The first 3 characters of a line can be ignored.

        # Print something to see the dictionary structure, etc.
        # print(f"Number of documents = {len(self.doc_set)}" + ".\n")
        # print(self.doc_set["3"]) # note that the dictionary indexes are strings, not numbers.

        with open('../cisi/CISI.QRY') as f:
            lines = ""
            for l in f.readlines():
                lines += "\n" + l.strip() if l.startswith(".") else " " + l.strip()
            lines = lines.lstrip("\n").split("\n")
        # qry_set = {}
        qry_id = ""
        for l in lines:
            if l.startswith(".I"):
                qry_id = l.split(" ")[1].strip()
            elif l.startswith(".W"):
                self.qry_set[qry_id] = l.strip()[3:]
                qry_id = ""

        # Print something to see the dictionary structure, etc.
        # print(f"Number of queries = {len(self.qry_set)}" + ".\n")
        # print(self.qry_set["3"])  # note that the dictionary indexes are strings, not numbers.
        # rel_set = {}
        with open('../cisi/CISI.REL') as f:
            for l in f.readlines():
                qry_id = l.lstrip(" ").strip("\n").split("\t")[0].split(" ")[0]
                doc_id = l.lstrip(" ").strip("\n").split("\t")[0].split(" ")[-1]
                if qry_id in self.rel_set:
                    self.rel_set[qry_id].append(doc_id)
                else:
                    self.rel_set[qry_id] = []
                    self.rel_set[qry_id].append(doc_id)
                # if qry_id == "7":
                #     print(l.strip("\n"))

        # Print something to see the dictionary structure, etc.
        # print(f"\nNumber of mappings = {len(self.rel_set)}" + ".\n")
        # print(self.rel_set["7"])  # note that the dictionary indexes are strings, not numbers.
def main():
    data = Data()
    data.readData()
main()