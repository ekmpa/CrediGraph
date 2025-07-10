import os
import re
from urllib.parse import urljoin, urlparse

import idna
from json_importer import json
from pyspark.sql.types import StringType, StructField, StructType,ArrayType
from sparkcc import CCSparkJob
# os.environ["PYSPARK_SUBMIT_ARGS"] = "--driver-memory MEM 4g"
from urllib.parse import urljoin, urlparse
from jsonpath_ng import jsonpath, parse as jsonpath_parse
import os
import shutil
class ExtractLinksJob(CCSparkJob):
    """Extract links from WAT files and redirects from WARC files
    and save them as pairs <from, to>.
    """

    name = 'ExtractLinks'

    output_schema = StructType(
        [StructField('url', StringType(), True), StructField('metadata', StringType(), True)]
    )
    metadataPaths_dict = {
        ###################### Generic MetaData ##############
        # "Request_Metadata": "Envelope.WARC-Header-Metadata",
        "MessageType": "Envelope.WARC-Header-Metadata.WARC-Type",  # can be [reques, respons, metadata]
        "URL": "Envelope.WARC-Header-Metadata.WARC-Target-URI",
        "Web_Domain_IP": "Envelope.WARC-Header-Metadata.WARC-IP-Address",
        "Content-Type": "Envelope.Payload-Metadata.Actual-Content-Type",
        "Request_Date": "Envelope.WARC-Header-Metadata.WARC-Date",
        ################# if request ###################
        "Request_Method": "Envelope.Payload-Metadata.HTTP-Request-Metadata.Request-Message.Method",  # [ Get or Post ]
        "Request_Path": "Envelope.Payload-Metadata.HTTP-Request-Metadata.Request-Message.Path",
        # [ link relative Path ]
        "Request_Accept-Language": "Envelope.Payload-Metadata.HTTP-Request-Metadata.Headers.Accept-Language",
        "Request_Host": "Envelope.Payload-Metadata.HTTP-Request-Metadata.Headers.Host",
        ################# if response ###################
        # "Response-Message": "Envelope.Payload-Metadata.HTTP-Response-Metadata.Response-Message",
        # "Response_Headers": "Envelope.Payload-Metadata.HTTP-Response-Metadata.Headers",
        "Content-language": "Envelope.Payload-Metadata.HTTP-Response-Metadata.Headers.content-language",
        "last-modified": "Envelope.Payload-Metadata.HTTP-Response-Metadata.Headers.last-modified",
        "Page_Title": "Envelope.Payload-Metadata.HTTP-Response-Metadata.HTML-Metadata.Head.Title",
        # "Page_Header_Links": "Envelope.Payload-Metadata.HTTP-Response-Metadata.HTML-Metadata.Head.Link",
        # "Page_Body_Links": "Envelope.Payload-Metadata.HTTP-Response-Metadata.HTML-Metadata.Links",
        # "Page_Script_Files": "Envelope.Payload-Metadata.HTTP-Response-Metadata.HTML-Metadata.Head.Scripts",
        "Page_Short_Desc": "Envelope.Payload-Metadata.HTTP-Response-Metadata.HTML-Metadata.Head.Metas.description",
        ################# if metadata ###################
        "cdl_languages": "Envelope.Payload-Metadata.WARC-Metadata-Metadata.Metadata-Records.languages-cld2"
        # Score per each language
    }
    warc_parse_http_header = False
    processing_robotstxt_warc = False
    records_response = None
    records_response_wat = None
    records_response_warc = None
    records_response_robotstxt = None
    records_failed = None
    records_non_html = None
    records_response_redirect = None
    link_count = None
    http_redirect_pattern = re.compile(b'^HTTP\\s*/\\s*1\\.[01]\\s*30[12378]\\b')
    http_redirect_location_pattern = re.compile(b'^Location:\\s*(\\S+)', re.IGNORECASE)
    http_link_pattern = re.compile(r'<([^>]*)>')
    http_success_pattern = re.compile(b'^HTTP\\s*/\\s*1\\.[01]\\s*200\\b')
    robotstxt_warc_path_pattern = re.compile(r'.*/robotstxt/')
    robotstxt_sitemap_pattern = re.compile(b'^Sitemap:\\s*(\\S+)', re.IGNORECASE)
    url_abs_pattern = re.compile(r'^(?:https?:)?//')

    # Meta properties usually offering links:
    #   <meta property="..." content="https://..." />
    html_meta_property_links = {
        'og:url',
        'og:image',
        'og:image:secure_url',
        'og:video',
        'og:video:url',
        'og:video:secure_url',
        'twitter:url',
        'twitter:image:src',
    }
    # Meta names usually offering links
    html_meta_links = {
        'twitter:image',
        'thumbnail',
        'application-url',
        'msapplication-starturl',
        'msapplication-TileImage',
        'vb_meta_bburl',
    }

    def add_arguments(self, parser):
        parser.add_argument(
            '--intermediate_output',
            type=str,
            default=None,
            help='Intermediate output to recover job from',
        )

    @staticmethod
    def _url_join(base, link):
        # TODO: efficiently join without reparsing base
        # TODO: canonicalize
        pass

    def iterate_records(self, warc_uri, archive_iterator):
        """Iterate over all WARC records and process them"""
        self.processing_robotstxt_warc = (
            ExtractLinksJob.robotstxt_warc_path_pattern.match(warc_uri)
        )
        count=0
        for record in archive_iterator:
            # if count<100:
            #     print("Processed Records Count:", self.records_response)
                for res in self.process_record(record):
                    yield res
                self.records_processed.add(1)
                # count+=1
            # else:
            #     break

    @staticmethod
    def parse_metadata(MetadataPaths_dict, json_record):
        metadata_dict = {}
        for k, v in MetadataPaths_dict.items():
            jsonpath_expression = jsonpath_parse(v)
            match = jsonpath_expression.find(json_record)
            if match:
                metadata_dict[k] = match[0].value
            else:
                metadata_dict[k] = ""
        metadata_dict["domain_name"] = urlparse(metadata_dict["URL"]).netloc
        return [metadata_dict]
    def process_record(self, record):
        if self.is_wat_json_record(record):
            try:
                wat_record = json.loads(self.get_payload_stream(record).read())
            except ValueError as e:
                self.get_logger().error('Failed to load JSON: {}'.format(e))
                self.records_failed.add(1)
                return []
            warc_header = wat_record['Envelope']['WARC-Header-Metadata']
            if warc_header['WARC-Type'] != 'response':
                # WAT request or metadata records
                return []

            self.records_response.add(1)
            self.records_response_wat.add(1)
            url = warc_header['WARC-Target-URI']
            for metadata_dict in self.parse_metadata(self.metadataPaths_dict, wat_record):
                meta_data= (str(url),  str(list(metadata_dict.values())))
                # print("meta_data=",meta_data)
                yield meta_data
        else:
            return []



    def init_accumulators(self, session):
        super(ExtractLinksJob, self).init_accumulators(session)

        sc = session.sparkContext
        self.records_failed = sc.accumulator(0)
        self.records_non_html = sc.accumulator(0)
        self.records_response = sc.accumulator(0)
        self.records_response_wat = sc.accumulator(0)
        self.records_response_warc = sc.accumulator(0)
        self.records_response_redirect = sc.accumulator(0)
        self.records_response_robotstxt = sc.accumulator(0)
        self.link_count = sc.accumulator(0)

    def log_accumulators(self, session):
        super(ExtractLinksJob, self).log_accumulators(session)

        self.log_accumulator(session, self.records_response, 'response records = {}')
        self.log_accumulator(
            session, self.records_failed, 'records failed to process = {}'
        )
        self.log_accumulator(session, self.records_non_html, 'records not HTML = {}')
        self.log_accumulator(
            session, self.records_response_wat, 'response records WAT = {}'
        )
        self.log_accumulator(
            session, self.records_response_warc, 'response records WARC = {}'
        )
        self.log_accumulator(
            session, self.records_response_redirect, 'response records redirects = {}'
        )
        self.log_accumulator(
            session, self.records_response_robotstxt, 'response records robots.txt = {}'
        )
        self.log_accumulator(session, self.link_count, 'non-unique link pairs = {}')


    def run_job(self, session):
        output = None
        out_path=str(session.conf.get("spark.sql.warehouse.dir")).split(":")[-1]+"/"+self.args.output
        if os.path.exists(out_path):
            shutil.rmtree(out_path)
        if self.args.input != '':
            input_data = session.sparkContext.textFile(
                self.args.input, minPartitions=self.args.num_input_partitions
            )
            output = input_data.mapPartitionsWithIndex(self.process_warcs)


        if not self.args.intermediate_output:
            df = session.createDataFrame(output, schema=self.output_schema)
        else:
            if output is not None:
                session.createDataFrame(output, schema=self.output_schema).write.format(
                    self.args.output_format
                ).option('compression', self.args.output_compression).saveAsTable(
                    self.args.intermediate_output
                )
                self.log_accumulators(session.sparkContext)
            warehouse_dir = session.conf.get(
                'spark.sql.warehouse.dir', 'spark-warehouse'
            )
            intermediate_output = os.path.join(
                warehouse_dir, self.args.intermediate_output
            )
            df = session.read.parquet(intermediate_output)

        df.dropDuplicates().coalesce(
            self.args.num_output_partitions
        ).sortWithinPartitions('url').write.format(self.args.output_format).option(
            'compression', self.args.output_compression
        ).mode("overwrite").saveAsTable(self.args.output)

        self.log_accumulators(session.sparkContext)


class ExtractHostLinksJob(ExtractLinksJob):
    """Extract links from WAT files, redirects from WARC files,
    and sitemap links from robots.txt response records.
    Extract the host names, reverse the names (example.com -> com.example)
    and save the pairs <source_host, target_host>.
    """

    name = 'ExtrHostLinks'
    output_schema = StructType(
            [StructField('url', StringType(), True), StructField('metadata', StringType(), True)]
        )
    num_input_partitions = 128
    num_output_partitions = 32

    # match global links
    # - with URL scheme, more restrictive than specified in
    #   https://tools.ietf.org/html/rfc3986#section-3.1
    # - or starting with //
    #   (all other "relative" links are within the same host)
    global_link_pattern = re.compile(
        r'^(?:[a-z][a-z0-9]{1,5}:)?//', re.IGNORECASE | re.ASCII
    )

    # match IP addresses
    # - including IPs with leading `www.' (stripped)
    ip_pattern = re.compile(r'^(?:www\.)?\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\Z')

    # valid host names, relaxed allowing underscore, allowing also IDNAs
    # https://en.wikipedia.org/wiki/Hostname#Restrictions_on_valid_hostnames
    host_part_pattern = re.compile(
        r'^[a-z0-9]([a-z0-9_-]{0,61}[a-z0-9])?\Z', re.IGNORECASE | re.ASCII
    )

    # simple pattern to match many but not all host names in URLs
    url_parse_host_pattern = re.compile(
        r'^https?://([a-z0-9_.-]{2,253})(?:[/?#]|\Z)', re.IGNORECASE | re.ASCII
    )






if __name__ == '__main__':
    job = ExtractHostLinksJob()
    job.run()
