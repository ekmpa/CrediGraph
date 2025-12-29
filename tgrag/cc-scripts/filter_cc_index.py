import pandas as pd
from pyspark.sql.types import IntegerType, StringType, StructField, StructType,ArrayType
from sparkcc import CCSparkJob
# os.environ["PYSPARK_SUBMIT_ARGS"] = "--driver-memory MEM 4g"
from urllib.parse import urljoin, urlparse
from jsonpath_ng import jsonpath, parse as jsonpath_parse
import os
import shutil
import requests
from pyspark.sql import SparkSession
from pyspark.sql import Window
from pyspark.sql.functions import row_number
from pyspark.sql import functions as F
import boto3
import botocore
from io import BytesIO
import pyarrow.parquet as pq
from tempfile import SpooledTemporaryFile, TemporaryFile
from pyspark.sql.functions import input_file_name
class Filter_CC_Index_Job(CCSparkJob):
    """Filter_CC_Index_Job .    """
    num_input_partitions = 64
    num_output_partitions = 4
    name = 'Filter_CC_Index_Job'
    cc_index_cols=['url_surtkey','url','url_host_name','url_host_tld','url_host_2nd_last_part','url_host_3rd_last_part','url_host_4th_last_part','url_host_5th_last_part','url_host_registry_suffix','url_host_registered_domain','url_host_private_suffix','url_host_private_domain','url_host_name_reversed','url_protocol','url_port','url_path','url_query','fetch_time','fetch_status','fetch_redirect','content_digest','content_mime_type','content_mime_detected','content_charset','content_languages','content_truncated','warc_filename','warc_record_offset','warc_record_length','warc_segment']
    index_toread_cols=['url','url_host_name', 'content_languages', 'warc_filename','warc_record_offset','warc_record_length','warc_segment']
    output_schema = StructType(
        [StructField('FileName', StringType(), True),
         StructField('url', StringType(), True),
         StructField('url_host_name', StringType(), True),
         StructField('content_languages', StringType(), True),
         StructField('warc_filename', StringType(), True),
         StructField('warc_record_offset', IntegerType(), True),
         StructField('warc_record_length', IntegerType(), True),
         StructField('warc_segment', StringType(), True),
         ]
    )
    parquet_records = None
    parquet_credibench_records = None
    records_failed = None
    domains_pc1_dict=None
    supported_langs = ["eng", "fra"] # "language codes: ISO-639-3 "
    filter_by_languages=False


    def add_arguments(self, parser):
        parser.add_argument(
            '--intermediate_output',
            type=str,
            default=None,
            help='Intermediate output to recover job from',
        )

    def fetch_parquet(self, uri, base_uri=None, offset=-1, length=-1):
        """Fetch WARC/WAT/WET files (or a record if offset and length are given)"""
        (scheme, netloc, path) = (None, None, None)
        uri_match = self.data_url_pattern.match(uri)
        if not uri_match and base_uri:
            # relative input URI (path) and base URI defined
            uri = base_uri + uri
            uri_match = self.data_url_pattern.match(uri)

        if uri_match:
            (scheme, netloc, path) = uri_match.groups()
        else:
            # keep local file paths as is
            path = uri

        stream = None

        if scheme == 's3':
            bucketname = netloc
            if not bucketname:
                self.get_logger().error('Invalid S3 URI: ' + uri)
                return
            if not path:
                self.get_logger().error('Empty S3 path: ' + uri)
                return
            elif path[0] == '/':
                # must strip leading / in S3 path
                path = path[1:]
            if offset > -1 and length > 0:
                rangereq = 'bytes={}-{}'.format(offset, (offset + length - 1))
                # Note: avoid logging too many small fetches
                # self.get_logger().debug('Fetching {} ({})'.format(uri, rangereq))
                try:
                    response = self.get_s3_client().get_object(
                        Bucket=bucketname, Key=path, Range=rangereq
                    )
                    stream = BytesIO(response['Body'].read())
                except botocore.client.ClientError as exception:
                    self.get_logger().error(
                        'Failed to download: s3://{}/{} (offset: {}, length: {}) - {}'.format(
                            bucketname, path, offset, length, exception
                        )
                    )
                    self.warc_input_failed.add(1)
                    return
            else:
                self.get_logger().info('Reading from S3 {}'.format(uri))
                # download entire file using a temporary file for buffering
                warctemp = TemporaryFile(mode='w+b', dir=self.args.local_temp_dir)
                try:
                    self.get_s3_client().download_fileobj(bucketname, path, warctemp)
                    warctemp.seek(0)
                    stream = warctemp
                except botocore.client.ClientError as exception:
                    self.get_logger().error(
                        'Failed to download {}: {}'.format(uri, exception)
                    )
                    self.warc_input_failed.add(1)
                    warctemp.close()

        elif scheme == 'http' or scheme == 'https':
            headers = None
            if offset > -1 and length > 0:
                headers = {'Range': 'bytes={}-{}'.format(offset, (offset + length - 1))}
                # Note: avoid logging many small fetches
                # self.get_logger().debug('Fetching {} ({})'.format(uri, headers))
            else:
                self.get_logger().info('Fetching {}'.format(uri))
            response = requests.get(uri, headers=headers)

            if response.ok:
                # includes "HTTP 206 Partial Content" for range requests
                warctemp = SpooledTemporaryFile(
                    max_size=2097152, mode='w+b', dir=self.args.local_temp_dir
                )
                warctemp.write(response.content)
                warctemp.seek(0)
                stream = warctemp
            else:
                self.get_logger().error(
                    'Failed to download {}: {}'.format(uri, response.status_code)
                )

        elif scheme == 'hdfs':
            try:
                import pydoop.hdfs as hdfs

                self.get_logger().error('Reading from HDFS {}'.format(uri))
                stream = hdfs.open(uri)
            except RuntimeError as exception:
                self.get_logger().error('Failed to open {}: {}'.format(uri, exception))
                self.warc_input_failed.add(1)

        else:
            self.get_logger().info('Reading local file {}'.format(uri))
            if scheme == 'file':
                # must be an absolute path
                uri = os.path.join('/', path)
            else:
                base_dir = os.path.abspath(os.path.dirname(__file__))
                uri = os.path.join(base_dir, uri)
            try:
                stream = open(uri, 'rb')
            except IOError as exception:
                self.get_logger().error('Failed to open {}: {}'.format(uri, exception))
                self.warc_input_failed.add(1)

        return stream

    def process_parquets(self, _id, iterator):
        """Process parquet files, calling iterate_records(...) for each file"""
        for uri in iterator:
            self.warc_input_processed.add(1)
            stream = self.fetch_parquet(uri, self.args.input_base_url)
            if not stream:
                continue
            for res in self.process_parquet(uri, stream):
                # print(f"res={res}")
                yield res
            stream.close()

    def process_parquet(self, uri, stream):
        """Parse a Parquet"""
        try:
            # print(f"uri={uri}")
            parquet_file = pq.ParquetFile(uri)
            for res in self.iterate_records(uri, parquet_file):
                yield res
        except Exception as exception:
            print(f"process_parquet exception={exception}")
            self.warc_input_failed.add(1)
            self.get_logger().error('Invalid Parquet: {} - {}'.format(uri, exception))


    def iterate_records(self, uri, parquet_file):
        file_name=uri.split("/cc-index/")[-1]
        # print(f"parquet_file={parquet_file}")
        chunk_size = 1e6
        for idx, batch in enumerate(parquet_file.iter_batches(batch_size=chunk_size, columns=self.index_toread_cols)):
            print(f"parquet {uri.split("/")[-1]} -> batch idx={idx}")
            # print(f"#parquet_records={self.parquet_records.value}")
            # print(f"#records_processed={self.records_processed.value}")
            chunk_df = batch.to_pandas()
            for row in chunk_df.itertuples():
                for res in self.process_record(row):
                    self.parquet_records.add(1)
                    url, url_host_name, content_languages, warc_filename, warc_record_offset, warc_record_length, warc_segment = res
                    if url_host_name:
                        self.records_processed.add(1)
                        yield (file_name,) + res

    def process_record(self, record):
        # print(f"parquet_records={record}")
        content_languages = record.content_languages
        self.parquet_credibench_records.add(1)
        if content_languages:
            content_languages_lst = content_languages.split(",")
            url,url_host_name,warc_filename,warc_record_offset,warc_record_length, warc_segment = None, None, None, None, None, None
            # print(f"self.args.filter_by_supported_languages={self.args.filter_by_supported_languages}")
            if not self.args.filter_by_supported_languages or (len(set(content_languages_lst) & set(self.supported_langs)) > 0):
                url_host_name = record.url_host_name
                if url_host_name in self.domains_pc1_dict.value:
                    url = record.url
                    warc_filename = record.warc_filename
                    warc_record_offset = record.warc_record_offset
                    warc_record_length = record.warc_record_length
                    warc_segment = record.warc_segment
                else:
                    url_host_name = None
            yield (url,url_host_name,content_languages,warc_filename,warc_record_offset,warc_record_length, warc_segment)
        return (None,None, None, None, None, None, None)

    def init_accumulators(self, session):
        super(Filter_CC_Index_Job, self).init_accumulators(session)
        sc = session.sparkContext
        self.records_failed = sc.accumulator(0)
        self.parquet_records = sc.accumulator(0)
        self.parquet_credibench_records = sc.accumulator(0)

    def log_accumulators(self, session):
        super(Filter_CC_Index_Job, self).log_accumulators(session)

        self.log_accumulator(session, self.parquet_records, 'parquet_records  = {}')
        self.log_accumulator(
            session, self.records_failed, 'records failed to process = {}'
        )
        # self.log_accumulator(session, self.records_non_html, 'records not HTML = {}')
        self.log_accumulator(
            session, self.parquet_credibench_records, 'parquet_credibench_records= {}'
        )
    @staticmethod
    def load_domain_pc1(domains_pc1_csv_path="../../data/dqr/domain_pc1.csv"):
        domains_df = pd.read_csv(domains_pc1_csv_path)
        return dict(zip(domains_df["domain"].tolist(), domains_df["pc1"].tolist()))
    def run_job(self, session):
        self.get_logger().info(f"seed domain path={self.args.trusted_domains}")
        out_path=str(session.conf.get("spark.sql.warehouse.dir")).split(":")[-1]+"/"+self.args.output
        self.domains_pc1_dict = session.sparkContext.broadcast(self.load_domain_pc1(self.args.trusted_domains))
        # print(f"out_path={out_path}")
        if os.path.exists(out_path):
            # print(f"Exists={out_path}")
            shutil.rmtree(out_path)
        if self.args.input != '':
            input_data = session.sparkContext.textFile(
            # input_data=session.sparkContext.wholeTextFiles(
                self.args.input, minPartitions=self.args.num_input_partitions
            )
            output = input_data.mapPartitionsWithIndex(self.process_parquets)
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
        df=df.dropDuplicates()
        df = df.filter(df.warc_record_length >= 500)
        w_desc = Window.partitionBy("url_host_name").orderBy(F.col("url_host_name").desc())
        w_asc = Window.partitionBy("url_host_name").orderBy(F.col("url_host_name").asc())
        df_low = (df
                  .withColumn("rn", F.row_number().over(w_asc))
                  .filter(F.col("rn") <= 3)
                  .drop("rn")
                  )
        df_high = (df
                   .withColumn("rn", F.row_number().over(w_desc))
                   .filter(F.col("rn") <= 3)
                   .drop("rn")
                   )
        df_final = df_low.union(df_high).distinct()
        df_final = df_final.orderBy(F.col("url_host_name").asc(), F.col("warc_record_length").asc())
        # df_final.collect()
        # df_sorted_Domain_Name_counts = df_final.groupBy("Domain_Name").count().collect()
        df_final.coalesce(self.args.num_output_partitions).write.format(self.args.output_format).option(
            'compression', self.args.output_compression
        ).mode("overwrite").saveAsTable(self.args.output)
        self.log_accumulators(session.sparkContext)
if __name__ == '__main__':
    job = Filter_CC_Index_Job()
    # ExtractWetContentsJob.domains_pc1_dict=load_domain_pc1()
    job.run()


