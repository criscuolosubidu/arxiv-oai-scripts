from lxml import etree
import pandas as pd
import os
import glob
from multiprocessing import Pool, cpu_count
import logging
import datetime
import traceback


def directfind(element,Xpath): #return text
    temp = element.find(Xpath)
    if temp is None:
        return ""
    text = temp.text
    if  text is None:
        return ""
    return text
def allfind(element,Xpath): #return
    temps = element.findall(Xpath)
    if temps is None:
        return ""
    temp_list =  []
    for temp in temps:
        text = temp.text
        if text is None:
            temp_list.append("")
        else:
            temp_list.append(text)
 
    return "/".join(temp_list)
 
def sub_allfind(element,Xpath):
    temps = element.findall(Xpath) #author Affliation
    if temps is None:
        return ""
    temp_list = []
    for temp in temps:
        text = temp.text
        if text is None:
            temp_list.append("")
        else:
            temp_list.append(text)
 
    return ";".join(temp_list)
 
def element_None_judge(element,Xpath):
    if element.find(Xpath) is None:
        return None
 
def get_pubdate(pubdate):
    Year = directfind(pubdate,"Year")
    Month = directfind(pubdate,"Month")
    Day = directfind(pubdate,"Day")
    return f'{Year}/{Month}/{Day}'
 
def get_authorname(Article):
    AuthorList = Article.find(".AuthorList")
    if AuthorList is None:
        return "","",""
    authors = AuthorList.findall(".Author") #Author (((LastName, ForeName?, Initials?, Suffix?) | CollectiveName), Identifier*, AffiliationInfo*) >
 
    name_list = []
    Initial_list = []
    Affiliations_list = [] #全部作者的机构
    for author in authors:
        LastName = directfind(author,'.//LastName')
        ForeName = directfind(author,'.//ForeName')
        Initials = directfind(author,'.//Initials')
 
        Affiliations  = sub_allfind(author,".//Affiliation")
        #作者名称进行拼接
 
        name_list.append(f'{LastName} {ForeName}')
        Initial_list.append(Initials)
        Affiliations_list.append(Affiliations)
 
 
    return "/".join(name_list),"/".join(Initial_list),"/".join(Affiliations_list)
 
def get_grant(Article):  #Grant (GrantID?, Acronym?, Agency, Country?)>
    GrantList = Article.find(".GrantList")
    if GrantList is None:
        return  "","",""
    grants = GrantList.findall(".Grant")
    GrantID_list = []
    Agency_list = []
    Country_list = []
    for grant in grants:
        GrantID = directfind(grant,'GrantID')
        Agency = directfind(grant,"Agency")
        Country = directfind(grant, "Country")
        GrantID_list.append(GrantID)
        Agency_list.append(Agency)
        Country_list.append(Country)
    return "/".join(GrantID_list), "/".join(Agency_list),"/".join(Country_list)
 
def get_PublicationType(Article):
    PublicationTypeList = Article.find(".PublicationTypeList")
    if PublicationTypeList is None:
        return ""
    return allfind(PublicationTypeList,"PublicationType")
 
def get_Chemical(MedlineCitation):
    ChemicalList = MedlineCitation.find(".ChemicalList")
    if ChemicalList is None:
        return "", ""
 
    RegistryNumber_list = []
    NameOfSubstance_list = []
    for Chemical in ChemicalList.findall(".Chemical"):
        RegistryNumber_list.append(directfind(Chemical,".RegistryNumber"))
        NameOfSubstance_list.append(directfind( Chemical,".NameOfSubstance"))
 
    return "/".join(RegistryNumber_list), "/".join(NameOfSubstance_list)
 
def get_MeshHeading(MedlineCitation):#MeshHeading (DescriptorName, QualifierName*)>
    MeshHeadingList = MedlineCitation.find(".MeshHeadingList")
    if MeshHeadingList is None:
        return "","","",""
    DescriptorName_list = []
    QualifierName_list = []
    for MeshHeading in  MeshHeadingList.findall( ".MeshHeading"): #可能仍然存在多个的情况
        DescriptorName_list.append(directfind(MeshHeading,".DescriptorName"))
        QualifierName_list.append(sub_allfind(MeshHeading,".QualifierName"))
 
    DescriptorNameMajorTopic = allfind(MeshHeadingList,".//DescriptorName[@MajorTopicYN='Y']")
    QualifierNameMajorTopic = allfind(MeshHeadingList,".//QualifierName[@MajorTopicYN='Y']")
 
    return "/".join(DescriptorName_list),"/".join(QualifierName_list), DescriptorNameMajorTopic,QualifierNameMajorTopic
 
def get_ArticleId(PubmedData):
    doi = directfind(PubmedData,".//ArticleId[@IdType='doi']")
    pubmedid = directfind(PubmedData,".//ArticleId[@IdType='pubmed']")
    return doi,pubmedid
 
def  get_Reference(PubmedData): #Reference (Citation, ArticleIdList?) >
    if PubmedData.find(".//Reference") is None:
        return "","",""
    References = PubmedData.findall(".//Reference") #找到所有reference
    Citation_list = []
    Citation_pubmedid_list = []
    Citation_doi_list = []
    for Reference in References:
        Citation_list.append(directfind(Reference,".Citation"))
        Citation_pubmedid_list.append(directfind(Reference,".//ArticleId[@IdType='pubmed']"))
        Citation_doi_list.append(directfind(Reference,".//ArticleId[@IdType='doi']"))
 
    return '/'.join(Citation_list),'/'.join(Citation_pubmedid_list),'/'.join(Citation_doi_list)
 
def get_Abstract(Article):  #Abstract？ Article
    if Article.find(".Abstract") is None:
        return ""
    Abstract =  Article.find(".Abstract")
    return  allfind(Abstract,".AbstractText")
 
def get_OtherAbstract(MedlineCitation):  #Abstract？ Article
    OtherAbstract = MedlineCitation.find(".//OtherAbstract")
    if OtherAbstract is None:
        return ""
    return allfind(OtherAbstract,".//AbstractText")
 
def get_Keyword (MedlineCitation): #KeywordList (Keyword+) >
    KeywordList = MedlineCitation.find(".KeywordList")
 
    if KeywordList is None:
        return "",""
    Keywords_MajorTopicY = allfind(MedlineCitation,".//Keyword[@MajorTopicYN = 'Y']")
    Keywords_MajorTopicN = allfind(MedlineCitation, ".//Keyword[@MajorTopicYN = 'N']")
 
 
    return Keywords_MajorTopicY,Keywords_MajorTopicN
 

def xml_to_csv(file, target):
    data = etree.parse(file)
    root = data.getroot()
    pubmedarticles = root.findall(".PubmedArticle")
 
    PMID_list = []
    pub_date_list = []
    Journal_Title_list = []
    ArticleTitle_list = []
    Authorname_list = []
    Initials_list = []
    Affiliation_list = []
    Language_list = []
    GrantID_list = []
    GrantAgency_list = []
    GrantCountry_list = []
    PublicationType_list = []
    Country_list = []
    MedlineTA_list = []
    NlmUniqueID_list = []
    ISSNLinking_list = []
    RegistryNumber_list = []
    NameOfSubstance_list = []
    DescriptorName_list = []
    QualifierName_list = []
    DescriptorNameMajorTopic_list = []
    QualifierNameMajorTopic_list = []
    Abstract_list = []
    OtherAbstract_list = []
    Keywords_MajorTopicY_list = []
    Keywords_MajorTopicN_list = []
    PublicationStatus_list = []
    pubmedid_list = []
    doi_list = []
    Citation_list = []
    Citatio_pumedid_list = []
    Citatio_doi_list = []
 
 
    for pubmedarticle in pubmedarticles:  # PubmedArticle (MedlineCitation, PubmedData?)>
        MedlineCitation = pubmedarticle.find(".MedlineCitation")
        '''
               MedlineCitation (PMID, DateCompleted?, DateRevised?, Article, 
                                        MedlineJournalInfo, ChemicalList?, SupplMeshList?,CitationSubset*, 
                                        CommentsCorrectionsList?, GeneSymbolList?, MeshHeadingList?, 
                                        NumberOfReferences?, PersonalNameSubjectList?, OtherID*, OtherAbstract*, 
                                        KeywordList*, CoiStatement?, SpaceFlightMission*, InvestigatorList*, GeneralNote*)>
        '''
        PMID = directfind(MedlineCitation,".PMID")
 
        Article = MedlineCitation.find(".Article")  # Article PubModel
        '''
               Article (Journal,ArticleTitle,((Pagination, ELocationID*) | ELocationID+),
                                Abstract?,AuthorList?, Language+, DataBankList?, GrantList?,
                                PublicationTypeList, VernacularTitle?, ArticleDate*) >
        '''
 
        Journal = Article.find(".Journal")  # Journal (ISSN?, JournalIssue, Title?, ISOAbbreviation?)>
        JournalIssue = Journal.find(".JournalIssue")
        pub_date = get_pubdate(JournalIssue.find(".PubDate"))
 
        Journal_Title =  directfind(Journal,".Title")
        ArticleTitle =directfind(Article,".ArticleTitle")
 
        Authorname, Initials, Affiliation = get_authorname(Article)
        Language =directfind(Article,".Language")
        GrantID, GrantAgency, GrantCountry = get_grant(Article)  # 不一定全部存在
 
        PublicationType = get_PublicationType(Article)
 
        MedlineJournalInfo = MedlineCitation.find(".MedlineJournalInfo")
        Country = directfind(MedlineJournalInfo,"Country")
        MedlineTA =directfind(MedlineJournalInfo,"MedlineTA")
        NlmUniqueID = directfind(MedlineJournalInfo,"NlmUniqueID")
        ISSNLinking = directfind(MedlineJournalInfo,"ISSNLinking")
 
        # ChemicalList
        RegistryNumber, NameOfSubstance = get_Chemical(MedlineCitation)
 
        # CitationSubset
 
        # MeshHeadingList
        DescriptorName, QualifierName, DescriptorNameMajorTopic, QualifierNameMajorTopic = get_MeshHeading(
            MedlineCitation)
 
        # Abstract？ Article
        Abstract = get_Abstract(Article)
 
        # OtherAbstract* MedlineCitation
        OtherAbstract = get_OtherAbstract(MedlineCitation)
 
        Keywords_MajorTopicY, Keywords_MajorTopicN = get_Keyword(MedlineCitation)
 
        PubmedData = pubmedarticle.find(".PubmedData")  # PubmedData (History?, PublicationStatus, ArticleIdList, ObjectList?, ReferenceList*) >
        if PubmedData is None:
            PublicationStatus = ""
            pubmedid = ""
            doi = ""
        else:
            PublicationStatus = PubmedData.find(".PublicationStatus").text
            doi, pubmedid = get_ArticleId(PubmedData)
 
        Citation, Citatio_pumedid, Citatio_doi = get_Reference(PubmedData)
        PMID_list.append(PMID)
        pub_date_list.append(pub_date)
        Journal_Title_list.append(Journal_Title)
        ArticleTitle_list.append(ArticleTitle)
        Authorname_list.append(Authorname)
        Initials_list.append(Initials)
        Affiliation_list.append(Affiliation)
        Language_list.append(Language)
        GrantID_list.append(GrantID)
        GrantAgency_list.append(GrantAgency)
        GrantCountry_list.append(GrantCountry)
        PublicationType_list.append(PublicationType)
        Country_list.append(Country)
        MedlineTA_list.append(MedlineTA)
        NlmUniqueID_list.append(NlmUniqueID)
        ISSNLinking_list.append(ISSNLinking)
        RegistryNumber_list.append(RegistryNumber)
        NameOfSubstance_list.append(NameOfSubstance)
        DescriptorName_list.append(DescriptorName)
        QualifierName_list.append(QualifierName)
        DescriptorNameMajorTopic_list.append(DescriptorNameMajorTopic)
        QualifierNameMajorTopic_list.append(QualifierNameMajorTopic)
        Abstract_list.append(Abstract)
        OtherAbstract_list.append(OtherAbstract)
        Keywords_MajorTopicY_list.append(Keywords_MajorTopicY)
        Keywords_MajorTopicN_list.append(Keywords_MajorTopicN)
        PublicationStatus_list.append(PublicationStatus)
        pubmedid_list.append(pubmedid)
        doi_list.append(doi)
        Citation_list.append(Citation)
        Citatio_pumedid_list.append(Citatio_pumedid)
        Citatio_doi_list.append(Citatio_doi)
 
    update_dict = {
        "PMID": PMID_list,
        "pub_date": pub_date_list,
        "Journal_Title": Journal_Title_list,
        "ArticleTitle": ArticleTitle_list,
        "Authorname": Authorname_list,
        "Initials": Initials_list,
        "Affiliation": Affiliation_list,
        "Language": Language_list,
        "GrantID": GrantID_list,
        "GrantAgency": GrantAgency_list,
        "GrantCountry": GrantCountry_list,
        "PublicationType": PublicationType_list,
        "Country": Country_list,
        "MedlineTA": MedlineTA_list,
        "NlmUniqueID": NlmUniqueID_list,
        "ISSNLinking": ISSNLinking_list,
        "RegistryNumber": RegistryNumber_list,
        "NameOfSubstance": NameOfSubstance_list,
        "DescriptorName": DescriptorName_list,
        "QualifierName": QualifierName_list,
        "DescriptorNameMajorTopic": DescriptorNameMajorTopic_list,
        "QualifierNameMajorTopic": QualifierNameMajorTopic_list,
        "Abstract": Abstract_list,
        "OtherAbstract": OtherAbstract_list,
        "Keywords_MajorTopicY": Keywords_MajorTopicY_list,
        "Keywords_MajorTopicN": Keywords_MajorTopicN_list,
        "PublicationStatus": PublicationStatus_list,
        "pubmedid": pubmedid_list,
        "doi": doi_list,
        "Citation": Citation_list,
        "Citatio_pumedid": Citatio_pumedid_list,
        "Citatio_doi": Citatio_doi_list,
    }
 
    data = pd.DataFrame(update_dict)
    data.to_excel(target, sheet_name= "Sheet1")
 
 
def setup_logging(log_dir="./logs"):
    """设置日志记录"""
    # 确保日志目录存在
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建带有时间戳的日志文件名
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"pubmed_process_{timestamp}.log")
    
    # 配置日志记录器
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()  # 同时输出到控制台
        ]
    )
    
    logging.info(f"日志已初始化，日志文件: {log_file}")
    return log_file

def xml_to_excel(input_xml_path, output_excel_path):
    """将单个XML文件转换为Excel文件"""
    try:
        logging.info(f"开始处理文件: {input_xml_path}")
        xml_to_csv(input_xml_path, output_excel_path)
        logging.info(f"处理完成: {input_xml_path} -> {output_excel_path}")
    except Exception as e:
        error_msg = traceback.format_exc()
        logging.error(f"处理文件 {input_xml_path} 时出错: {str(e)}\n{error_msg}")
        
def process_xml_files(xml_dir, excel_dir, num_processes=16):
    """使用多进程处理指定目录下的所有XML文件"""
    # 确保输出目录存在
    os.makedirs(excel_dir, exist_ok=True)
    
    # 获取所有XML文件
    xml_files = glob.glob(os.path.join(xml_dir, "*.xml"))
    
    if not xml_files:
        logging.warning(f"在目录 {xml_dir} 中没有找到XML文件")
        return
    
    # 准备参数列表
    tasks = []
    for xml_file in xml_files:
        filename = os.path.basename(xml_file)
        output_file = os.path.join(excel_dir, filename.replace('.xml', '.xlsx'))
        tasks.append((xml_file, output_file))
    
    # 使用进程池处理文件
    logging.info(f"开始使用 {num_processes} 个进程处理 {len(xml_files)} 个XML文件...")
    with Pool(processes=num_processes) as pool:
        pool.starmap(xml_to_excel, tasks)
    
    logging.info(f"所有文件处理完成，结果已保存到 {excel_dir}")

if __name__ == '__main__':
    # 设置日志
    log_file = setup_logging()
    
    xml_directory = "./data/xml"
    excel_directory = "./data/xlsx"
    num_processes = 16
    
    try:
        logging.info(f"启动程序，处理目录: {xml_directory}，输出到: {excel_directory}")
        process_xml_files(xml_directory, excel_directory, num_processes)
        logging.info("程序正常结束")
    except Exception as e:
        error_msg = traceback.format_exc()
        logging.critical(f"程序执行过程中发生严重错误: {str(e)}\n{error_msg}")

