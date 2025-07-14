import os
import re
from urllib.parse import urljoin, urlparse
import pandas as pd
import idna
from json_importer import json
from pyspark.sql.types import IntegerType, StringType, StructField, StructType,ArrayType
from sparkcc import CCSparkJob
# os.environ["PYSPARK_SUBMIT_ARGS"] = "--driver-memory MEM 4g"
from urllib.parse import urljoin, urlparse
from jsonpath_ng import jsonpath, parse as jsonpath_parse
import os
import shutil
class ExtractWetContentsJob(CCSparkJob):
    """Extract links from WAT files and redirects from WARC files
    and save them as pairs <from, to>.
    """
    num_input_partitions = 128
    num_output_partitions = 2
    name = 'ExtractWetContentsJob'
    output_schema = StructType(
        [StructField('Domain_Name', StringType(), True),
         StructField('WARC_Target_URI', StringType(), True),
         StructField('WARC_Identified_Content_Language', StringType(), True),
         StructField('WARC_Date', StringType(), True),
         StructField('Content_Type', StringType(), True),
         StructField('Content_Length', IntegerType(), True),
         StructField('wet_record_txt', StringType(), True),
         ]
    )
    records_response = None
    records_response_wet = None
    records_failed = None
    domains_pc1_dict=None
    supported_langs = ["eng", "fra"] # "language codes: ISO-639-3 "


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
        count=0
        for record in archive_iterator:
                for res in self.process_record(record):
                    Domain_Name, WARC_Target_URI, WARC_Identified_Content_Language, WARC_Date, Content_Type,Content_Length, wet_record_txt=res
                    if Domain_Name:
                        yield res
                self.records_processed.add(1)

    def process_record(self, record):
        self.records_response.add(1)
        if self.is_wet_text_record(record):
            self.records_response_wet.add(1)
            WARC_Identified_Content_Language = record.rec_headers['WARC-Identified-Content-Language']
            if WARC_Identified_Content_Language:
                WARC_Identified_Content_Languages_lst = WARC_Identified_Content_Language.split(",")
                Domain_Name, WARC_Target_URI, WARC_Date, Content_Type, Content_Length, wet_record_txt = None, None, None, None, None, None

                if len(set(WARC_Identified_Content_Languages_lst) & set(self.supported_langs)) > 0:
                    WARC_Target_URI = record.rec_headers['WARC-Target-URI']
                    Domain_Name = urlparse(WARC_Target_URI).netloc
                    if Domain_Name in self.domains_pc1_dict.value:
                        WARC_Date = record.rec_headers['WARC-Date']
                        Content_Type = record.rec_headers['Content-Type']
                        Content_Length = int(record.rec_headers['Content-Length'])
                        wet_record_txt = self.get_payload_stream(record).read().decode('utf-8')
                    else:
                        Domain_Name = None
                yield (Domain_Name, WARC_Target_URI, WARC_Identified_Content_Language, WARC_Date, Content_Type, Content_Length, wet_record_txt)
            return (None, None, None, None, None, None, None)
        else:
            return (None, None, None, None, None, None, None)

    def init_accumulators(self, session):
        super(ExtractWetContentsJob, self).init_accumulators(session)
        sc = session.sparkContext
        self.records_failed = sc.accumulator(0)
        # self.records_non_html = sc.accumulator(0)
        self.records_response = sc.accumulator(0)
        self.records_response_wet = sc.accumulator(0)
        self.records_response_warc = sc.accumulator(0)
        # self.records_response_redirect = sc.accumulator(0)
        # self.records_response_robotstxt = sc.accumulator(0)
        # self.link_count = sc.accumulator(0)

    def log_accumulators(self, session):
        super(ExtractWetContentsJob, self).log_accumulators(session)

        self.log_accumulator(session, self.records_response, 'response records = {}')
        self.log_accumulator(
            session, self.records_failed, 'records failed to process = {}'
        )
        # self.log_accumulator(session, self.records_non_html, 'records not HTML = {}')
        self.log_accumulator(
            session, self.records_response_wet, 'response records WET = {}'
        )

        # self.log_accumulator(
        #     session, self.records_response_redirect, 'response records redirects = {}'
        # )
    @staticmethod
    def load_domain_pc1(domains_pc1_csv_path="../../data/dqr/domain_pc1.csv"):
        doamins_df = pd.read_csv(domains_pc1_csv_path)
        return dict(zip(doamins_df["domain"].tolist(), doamins_df["pc1"].tolist()))
    def run_job(self, session):
        out_path=str(session.conf.get("spark.sql.warehouse.dir")).split(":")[-1]+"/"+self.args.output
        self.domains_pc1_dict = session.sparkContext.broadcast(self.load_domain_pc1(self.args.trusted_domains))
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
        ).sortWithinPartitions('Domain_Name',"WARC_Target_URI").write.format(self.args.output_format).option(
            'compression', self.args.output_compression
        ).mode("overwrite").saveAsTable(self.args.output)

        self.log_accumulators(session.sparkContext)




if __name__ == '__main__':
    job = ExtractWetContentsJob()
    # ExtractWetContentsJob.domains_pc1_dict=load_domain_pc1()
    job.run()


