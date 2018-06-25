/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// See docs in ../ops/parsing_ops.cc.
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/lib/io/random_inputstream.h"

namespace tensorflow {
namespace {

class CSVDatasetOp : public DatasetOpKernel {
 public:
  explicit CSVDatasetOp(OpKernelConstruction* ctx) : DatasetOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_types", &output_types_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_shapes", &output_shapes_));
  }

  void MakeDataset(OpKernelContext* ctx, DatasetBase** output) override {
    const Tensor* filenames_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("filenames", &filenames_tensor));
    OP_REQUIRES(
        ctx, filenames_tensor->dims() <= 1,
        errors::InvalidArgument("`filenames` must be a scalar or a vector."));

    OpInputList record_defaults_list;
    OP_REQUIRES_OK(ctx,
                   ctx->input_list("record_defaults", &record_defaults_list));
    for (int i = 0; i < record_defaults_list.size(); ++i) {
      OP_REQUIRES(ctx, record_defaults_list[i].NumElements() < 2,
                  errors::InvalidArgument(
                      "There should only be 1 default per field but field ", i,
                      " has ", record_defaults_list[i].NumElements()));
    }

    const Tensor* select_cols_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("select_cols", &select_cols_tensor));
    OP_REQUIRES(ctx, select_cols_tensor->dims() == 1,
                errors::InvalidArgument("`select_cols` must be a vector."));

    int64 buffer_size;
    OP_REQUIRES_OK(
        ctx, ParseScalarArgument<int64>(ctx, "buffer_size", &buffer_size));
    OP_REQUIRES(ctx, buffer_size > 0,
                errors::InvalidArgument("buffer_size should be positive"));

    string delim;
    OP_REQUIRES_OK(ctx,
                   ParseScalarArgument<string>(ctx, "field_delim", &delim));
    OP_REQUIRES(ctx, delim.size() == 1,
                errors::InvalidArgument("field_delim should be only 1 char"));

    bool header;
    OP_REQUIRES_OK(ctx, ParseScalarArgument<bool>(ctx, "header", &header));

    bool use_quote_delim;
    OP_REQUIRES_OK(ctx, ParseScalarArgument<bool>(ctx, "use_quote_delim",
                                                  &use_quote_delim));
    string na_value;
    OP_REQUIRES_OK(ctx,
                   ParseScalarArgument<string>(ctx, "na_value", &na_value));

    std::vector<Tensor> record_defaults;
    record_defaults.reserve(record_defaults_list.size());
    for (const Tensor& t : record_defaults_list) {
      record_defaults.push_back(t);
    }

    std::vector<string> filenames;
    filenames.reserve(filenames_tensor->NumElements());
    for (int i = 0; i < filenames_tensor->NumElements(); ++i) {
      filenames.push_back(filenames_tensor->flat<string>()(i));
    }

    std::vector<int64> select_cols;
    select_cols.reserve(select_cols_tensor->NumElements());
    for (int i = 0; i < select_cols_tensor->NumElements(); ++i) {
      select_cols.push_back(select_cols_tensor->flat<int64>()(i));
    }
    OP_REQUIRES(
        ctx, output_types_.size() == select_cols.size() || select_cols.empty(),
        errors::InvalidArgument("select_cols should match output size"));
    for (int i = 1; i < select_cols.size(); i++) {
      OP_REQUIRES(ctx, select_cols[i - 1] < select_cols[i],
                  errors::InvalidArgument(
                      "select_cols should be strictly increasing indices"));
    }
    OP_REQUIRES(
        ctx, select_cols.empty() || select_cols.front() >= 0,
        errors::InvalidArgument("select_cols should be non-negative indices"));

    *output = new Dataset(ctx, std::move(filenames), header, buffer_size,
                          output_types_, output_shapes_,
                          std::move(record_defaults), std::move(select_cols),
                          use_quote_delim, delim[0], std::move(na_value));
  }

 private:
  class Dataset : public GraphDatasetBase {
   public:
    Dataset(OpKernelContext* ctx, std::vector<string> filenames, bool header,
            int64 buffer_size, const DataTypeVector& output_types,
            const std::vector<PartialTensorShape>& output_shapes,
            std::vector<Tensor> record_defaults, std::vector<int64> select_cols,
            bool use_quote_delim, char delim, string na_value)
        : GraphDatasetBase(ctx),
          filenames_(std::move(filenames)),
          header_(header),
          buffer_size_(buffer_size),
          out_type_(output_types),
          output_shapes_(output_shapes),
          record_defaults_(std::move(record_defaults)),
          select_cols_(std::move(select_cols)),
          use_quote_delim_(use_quote_delim),
          delim_(delim),
          na_value_(std::move(na_value)) {}

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      return std::unique_ptr<IteratorBase>(
          new Iterator({this, strings::StrCat(prefix, "::CSV")}));
    }

    const DataTypeVector& output_dtypes() const override { return out_type_; }

    const std::vector<PartialTensorShape>& output_shapes() const override {
      return output_shapes_;
    }

    string DebugString() const override { return "CSVDatasetOp::Dataset"; }

   protected:
    Status AsGraphDefInternal(DatasetGraphDefBuilder* b,
                              Node** output) const override {
      // TODO(rachelim): Implement this
      std::vector<Node*> input_tensors;
      TF_RETURN_IF_ERROR(b->AddDataset(this, input_tensors, output));
      return errors::Unimplemented("CSVDataset: AsGraphDefInternal");
    }

   private:
    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Params& params)
          : DatasetIterator<Dataset>(params) {}

      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
        mutex_lock l(mu_);
        bool select_all = dataset()->select_cols_.empty();
        do {
          // We are currently processing a file, so try to read the next record
          if (input_stream_) {
            Status s = ReadRecord(ctx, out_tensors, select_all,
                                  dataset()->select_cols_);
            if (s.ok()) {
              // Validate output
              if (out_tensors->size() != dataset()->out_type_.size()) {
                return errors::InvalidArgument(
                    "Expect ", dataset()->out_type_.size(), " fields but have ",
                    out_tensors->size(), " in record");
              }

              *end_of_sequence = false;
              return s;
            }
            if (!errors::IsOutOfRange(s)) {
              // Not at the end of file, return OK or non-EOF errors to caller.
              *end_of_sequence = false;
              return s;
            }
            // We have reached the end of the current file, so maybe
            // move on to next file.
            ResetStreamsLocked();
            ++current_file_index_;
          }
          // Iteration ends when there are no more files to process.
          if (current_file_index_ == dataset()->filenames_.size()) {
            *end_of_sequence = true;
            return Status::OK();
          }
          TF_RETURN_IF_ERROR(SetupStreamsLocked(ctx->env()));
        } while (true);
      }

     protected:
      Status SaveInternal(IteratorStateWriter* writer) override {
        mutex_lock l(mu_);
        // TODO(rachelim): Implement save
        return errors::Unimplemented("CSVDataset: SaveInternal");
      }
      Status RestoreInternal(IteratorContext* ctx,
                             IteratorStateReader* reader) override {
        mutex_lock l(mu_);
        // TODO(rachelim): Implement restore
        return errors::Unimplemented("CSVDataset: RestoreInternal");
      }

     private:
      // Reads an entire CSV row from the input stream, either from the
      // existing buffer or by filling the buffer as needed. Converts extracted
      // fields to output tensors as we go.
      //
      // When this function is called, pos_ should be the index of the first
      // character of the record in buffer_, or past the end of the buffer.
      // Note: ctx and out_tensors are only used in this function
      // when fields are included in the record.
      Status ReadRecord(IteratorContext* ctx, std::vector<Tensor>* out_tensors,
                        bool select_all, const std::vector<int64>& selected)
          EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        if (pos_ >= buffer_.size()) {
          // At the end of the file, this will return errors::OutOfRange
          TF_RETURN_IF_ERROR(FillBuffer(&buffer_));
          pos_ = 0;
        }

        // The first character may be \n if this is the continuation of a
        // \r\n linebreak between this and the previous record. If so, skip it.

        bool end_of_record = false;  // Keep track of when we find \n, \r or EOF
        size_t num_parsed = 0;
        size_t num_selected_parsed = 0;

        Status result;

        while (!end_of_record) {  // Read till we reach \n, \r or EOF
          bool include =
              select_all || (num_selected_parsed < selected.size() &&
                             selected[num_selected_parsed] == num_parsed);

          // Don't fail fast, so that the next call to GetNext may still return
          // a valid record
          result.Update(
              ParseOneField(ctx, out_tensors, &end_of_record, include));

          num_parsed++;
          if (include) num_selected_parsed++;
        }

        return result;
      }

      // Parses one field from position pos_ in the buffer. Fields are
      // delimited by delim, CRLF, or EOF. Advances pos_ to the first char of
      // the next field.
      Status ParseOneField(IteratorContext* ctx,
                           std::vector<Tensor>* out_tensors,
                           bool* end_of_record, bool include)
          EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        if (pos_ >= buffer_.size()) {
          // If we get here, this means the previous field's end coincided
          // with the end of the buffer. We can fill the buffer without abandon.
          Status s = FillBuffer(&buffer_);

          if (errors::IsOutOfRange(s)) {
            // Reached EOF, and last field is empty
            *end_of_record = true;
            if (include) {
              return FieldToOutput(ctx, StringPiece(), out_tensors);
            } else {
              return Status::OK();
            }
          } else if (!s.ok()) {
            return s;  // Surface other errors back to caller
          }

          pos_ = 0;
        }

        if (dataset()->use_quote_delim_ && buffer_[pos_] == '"') {
          return ParseQuotedField(ctx, out_tensors, end_of_record, include);
        }

        return ParseUnquotedField(ctx, out_tensors, end_of_record, include);
      }

      // For keeping track of relevant parts of a field from a previous buffer
      struct Piece {
        size_t start;
        size_t len;
        string buffer;

        Piece(string buffer, size_t start, size_t len)
            : start(start), len(len), buffer(std::move(buffer)) {}
      };

      // Given that pos_ exceeds the buffer, saves the relevant part of the
      // current buffer (if necessary), fills the buffer, and resets indices to
      // 0.
      Status SaveAndFillBuffer(std::vector<Piece>* earlier_pieces,
                               size_t* start, bool include)
          EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        string temp_buffer;

        buffer_.swap(temp_buffer);
        if (include && pos_ > *start) {
          earlier_pieces->push_back(
              Piece(std::move(temp_buffer), *start, pos_ - *start));
        }
        pos_ = 0;
        *start = 0;
        return FillBuffer(&buffer_);
      }

      // Parses unquoted field from position pos_ in the buffer. Continually
      // reads from buffer until end of field is reached (delim, CRLF, or EOF).
      // Advances pos_ to keep track of our position in the buffer as we go,
      // stopping at the first character of the next field.
      Status ParseQuotedField(IteratorContext* ctx,
                              std::vector<Tensor>* out_tensors,
                              bool* end_of_record, bool include)
          EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        std::vector<Piece> earlier_pieces;
        size_t start = pos_;
        pos_++;  // Starting quotation mark

        Status parse_result;
        while (true) {  // Each iter reads 1 char, filling buffer if necessary
          if (pos_ >= buffer_.size()) {
            Status s = SaveAndFillBuffer(&earlier_pieces, &start, include);
            if (errors::IsOutOfRange(s)) {
              return errors::InvalidArgument(
                  "Reached end of file without closing quoted field in "
                  "record");
            } else if (!s.ok()) {
              return s;  // Surface all other errors to caller
            }
          }

          char ch = buffer_[pos_];
          if (ch == '"') {
            // When we encounter a quote, we look ahead to the next character to
            // decide what to do
            pos_++;
            if (pos_ >= buffer_.size()) {
              Status s = SaveAndFillBuffer(&earlier_pieces, &start, include);
              if (errors::IsOutOfRange(s)) {
                // This was the last field. We are done
                *end_of_record = true;
                parse_result.Update(QuotedFieldToOutput(
                    ctx, StringPiece(), out_tensors, earlier_pieces, include));
                return parse_result;
              } else if (!s.ok()) {
                return s;
              }
            }

            char next = buffer_[pos_];
            pos_++;
            if (next == dataset()->delim_) {
              parse_result.Update(QuotedFieldToOutput(
                  ctx, StringPiece(&buffer_[start], pos_ - 1 - start),
                  out_tensors, earlier_pieces, include));
              return parse_result;

            } else if (next == '\n' || next == '\r') {
              *end_of_record = true;
              parse_result.Update(QuotedFieldToOutput(
                  ctx, StringPiece(&buffer_[start], pos_ - 1 - start),
                  out_tensors, earlier_pieces, include));
              if (next == '\r') SkipNewLineIfNecessary();
              return parse_result;
            } else if (next != '"') {
              // Take note of the error, but keep going to end of field.
              include = false;  // So we don't get funky errors when trying to
                                // unescape the quotes.
              parse_result.Update(errors::InvalidArgument(
                  "Quote inside a string has to be escaped by another quote"));
            }

          } else {
            pos_++;
          }
        }
      }

      // Converts quoted field to an output tensor, removing the starting
      // and ending quotes from it and unescaping double quotations if
      // necessary.
      Status QuotedFieldToOutput(IteratorContext* ctx, StringPiece field,
                                 std::vector<Tensor>* out_tensors,
                                 const std::vector<Piece>& earlier_pieces,
                                 bool include) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        if (!include) return Status::OK();

        if (earlier_pieces.empty()) {
          if (field.find('\"', 1) == field.size() - 1) {
            // `field` contains no escaped quotation marks.
            // Exclude framing quotation marks
            field.remove_prefix(1);
            field.remove_suffix(1);
            return FieldToOutput(ctx, field, out_tensors);
          }
        }
        string field_complete;
        size_t str_len = field.size();
        for (const Piece& p : earlier_pieces) {
          str_len += p.len;
        }
        field_complete.reserve(str_len);

        // This bool flips every time we see a quote, so that we skip the second
        // quote of every pair of adjacent quotes in the field. We need to track
        // this across iterations of the for loop because adjacent double quotes
        // may be in different buffers. Initialize to true because we also skip
        // the opening quotation mark of the quoted field.
        bool skip_next_quote = true;
        for (const Piece& p : earlier_pieces) {
          AppendUnescapedPiece(StringPiece(&p.buffer[p.start], p.len),
                               &field_complete, &skip_next_quote);
        }
        AppendUnescapedPiece(field, &field_complete, &skip_next_quote);
        StringPiece result = StringPiece(field_complete);
        result.remove_suffix(1);  // Skip final quote

        return FieldToOutput(ctx, result, out_tensors);
      }

      void AppendUnescapedPiece(StringPiece piece, string* field_complete,
                                bool* skip_next_quote) {
        size_t from = 0;
        size_t found = piece.find('\"', from);
        while (found != string::npos) {
          if (!*skip_next_quote) {
            // This is the first quote in a pair of adjacent double quotes
            field_complete->append(piece.data() + from, found + 1 - from);
          }
          *skip_next_quote = !*skip_next_quote;
          from = found + 1;
          found = piece.find('\"', from);
        }
        // Include the chunk after the last quotation mark in the string
        if (from < piece.size()) {
          field_complete->append(piece.data() + from, piece.size() - from);
        }
      }

      // Parses unquoted field from position pos_ in the buffer. Continually
      // reads from buffer until end of field is reached (delim, CRLF, or EOF).
      // Advances pos_ to keep track of our position in the buffer as we go,
      // stopping at the first character of the next field.
      Status ParseUnquotedField(IteratorContext* ctx,
                                std::vector<Tensor>* out_tensors,
                                bool* end_of_record, bool include)
          EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        std::vector<Piece> earlier_pieces;
        size_t start = pos_;
        Status parse_result;

        while (true) {  // Each iter reads 1 char, filling buffer if necessary
          if (pos_ >= buffer_.size()) {
            Status s = SaveAndFillBuffer(&earlier_pieces, &start, include);
            // Handle errors
            if (errors::IsOutOfRange(s)) {
              // Whatever we have is the last field of the last record
              *end_of_record = true;
              parse_result.Update(UnquotedFieldToOutput(
                  ctx, StringPiece(&buffer_[start], pos_ - start), out_tensors,
                  earlier_pieces, include));
              return parse_result;
            } else if (!s.ok()) {
              return s;  // Surface all other errors to caller
            }
          }

          char ch = buffer_[pos_];

          if (ch == dataset()->delim_) {
            parse_result.Update(UnquotedFieldToOutput(
                ctx, StringPiece(&buffer_[start], pos_ - start), out_tensors,
                earlier_pieces, include));
            pos_++;
            return parse_result;
          }
          if (ch == '\n' || ch == '\r') {
            // need special case to skip over first \n of record if the line
            // breaks are \r\n
            parse_result.Update(UnquotedFieldToOutput(
                ctx, StringPiece(&buffer_[start], pos_ - start), out_tensors,
                earlier_pieces, include));
            *end_of_record = true;
            pos_++;
            if (ch == '\r') SkipNewLineIfNecessary();
            return parse_result;
          }
          if (dataset()->use_quote_delim_ && ch == '"') {
            // Take note of the error, but keep going to end of field.
            parse_result.Update(errors::InvalidArgument(
                "Unquoted fields cannot have quotes inside"));
          }
          // Otherwise, go to next character
          pos_++;
        }
      }

      Status FillBuffer(string* result) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        result->clear();
        Status s = input_stream_->ReadNBytes(dataset()->buffer_size_, result);

        if (errors::IsOutOfRange(s) && !result->empty()) {
          // Ignore OutOfRange error when ReadNBytes read < N bytes.
          return Status::OK();
        }
        return s;
      }

      // Given a field, converts it to the right output tensor type
      Status FieldToOutput(IteratorContext* ctx, StringPiece field,
                           std::vector<Tensor>* out_tensors) {
        size_t output_idx = out_tensors->size();
        if (output_idx >= dataset()->out_type_.size()) {
          // We can get here if we're selecting all columns, but the number of
          // fields exceeds the number of defaults provided
          return errors::InvalidArgument("Expect ", dataset()->out_type_.size(),
                                         " fields but have more in record");
        }
        const DataType& dtype = dataset()->out_type_[output_idx];
        Tensor component(ctx->allocator({}), dtype, {});
        if ((field.empty() || field == dataset()->na_value_) &&
            dataset()->record_defaults_[output_idx].NumElements() != 1) {
          // If the field is empty or NA value, and default is not given,
          // report error.
          return errors::InvalidArgument("Field ", output_idx,
                                         " is required but missing in record!");
        }

        switch (dtype) {
          // For each case, if the field is empty, we use the default.
          // Otherwise, we convert it to the right type.
          case DT_INT32: {
            if (field.empty() || field == dataset()->na_value_) {
              component.scalar<int32>()() =
                  dataset()->record_defaults_[output_idx].flat<int32>()(0);
            } else {
              int32 value;
              if (!strings::safe_strto32(field, &value)) {
                return errors::InvalidArgument(
                    "Field ", output_idx,
                    " in record is not a valid int32: ", field);
              }
              component.scalar<int32>()() = value;
            }
            break;
          }
          case DT_INT64: {
            if (field.empty() || field == dataset()->na_value_) {
              component.scalar<int64>()() =
                  dataset()->record_defaults_[output_idx].flat<int64>()(0);
            } else {
              int64 value;
              if (!strings::safe_strto64(field, &value)) {
                return errors::InvalidArgument(
                    "Field ", output_idx,
                    " in record is not a valid int64: ", field);
              }
              component.scalar<int64>()() = value;
            }
            break;
          }
          case DT_FLOAT: {
            if (field.empty() || field == dataset()->na_value_) {
              component.scalar<float>()() =
                  dataset()->record_defaults_[output_idx].flat<float>()(0);
            } else {
              float value;
              if (!strings::safe_strtof(field, &value)) {
                return errors::InvalidArgument(
                    "Field ", output_idx,
                    " in record is not a valid float: ", field);
              }
              component.scalar<float>()() = value;
            }
            break;
          }
          case DT_DOUBLE: {
            if (field.empty() || field == dataset()->na_value_) {
              component.scalar<double>()() =
                  dataset()->record_defaults_[output_idx].flat<double>()(0);
            } else {
              double value;
              if (!strings::safe_strtod(field, &value)) {
                return errors::InvalidArgument(
                    "Field ", output_idx,
                    " in record is not a valid double: ", field);
              }
              component.scalar<double>()() = value;
            }
            break;
          }
          case DT_STRING: {
            if (field.empty() || field == dataset()->na_value_) {
              component.scalar<string>()() =
                  dataset()->record_defaults_[output_idx].flat<string>()(0);
            } else {
              component.scalar<string>()() = field.ToString();
            }
            break;
          }
          default:
            return errors::InvalidArgument("csv: data type ", dtype,
                                           " not supported in field ",
                                           output_idx);
        }
        out_tensors->push_back(std::move(component));
        return Status::OK();
      }

      // Records can be delimited by "\r\n" line breaks. When we encounter a
      // '\r', we have to check the next character to see if it is part of the
      // linebreak, and ignore it if so.
      void SkipNewLineIfNecessary() EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        if (pos_ >= buffer_.size()) {
          Status s = FillBuffer(&buffer_);
          pos_ = 0;
          // If we failed to fill buffer, it doesn't matter because we're done
          // with the record
          if (!s.ok()) return;
        }
        if (buffer_[pos_] == '\n') {
          pos_++;
        }
      }

      // Given a string field, and its index in the output,
      // converts it to a Tensor of the right type and adds it to the
      // out_tensors vector.
      Status UnquotedFieldToOutput(IteratorContext* ctx, StringPiece field,
                                   std::vector<Tensor>* out_tensors,
                                   const std::vector<Piece>& earlier_pieces,
                                   bool include) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        if (!include) return Status::OK();

        if (earlier_pieces.empty()) {
          return FieldToOutput(ctx, field, out_tensors);
        }

        size_t str_len = field.size();
        for (const Piece& p : earlier_pieces) {
          str_len += p.len;
        }
        string field_complete;
        field_complete.reserve(str_len);

        for (const Piece& p : earlier_pieces) {
          field_complete.append(p.buffer, p.start, p.len);
        }

        field_complete.append(field.data(), field.size());
        return FieldToOutput(ctx, field_complete, out_tensors);
      }

      // Sets up reader streams to read from the file at `current_file_index_`.
      Status SetupStreamsLocked(Env* env) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        if (current_file_index_ >= dataset()->filenames_.size()) {
          return errors::InvalidArgument(
              "current_file_index_:", current_file_index_,
              " >= filenames_.size():", dataset()->filenames_.size());
        }

        // Actually move on to next file.
        TF_RETURN_IF_ERROR(env->NewRandomAccessFile(
            dataset()->filenames_[current_file_index_], &file_));
        input_stream_.reset(
            new io::RandomAccessInputStream(file_.get(), false));
        buffer_.clear();
        pos_ = 0;
        if (dataset()->header_) {
          // Read one line, but don't include it. Pass nullptrs as dummy
          // pointers to objects that shouldn't be invoked anyway
          // We need to process this as a record here instead of just finding
          // the first newline because it might contain quoted fields with
          // newlines in the header as well
          std::vector<int64> empty;
          Status s = ReadRecord(nullptr, nullptr, false, empty);
          if (!s.ok()) {
            return errors::InvalidArgument("Can't read header of file");
          }
        }
        return Status::OK();
      }

      // Resets all reader streams.
      void ResetStreamsLocked() EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        input_stream_.reset();
        file_.reset();
      }

      mutex mu_;
      string buffer_ GUARDED_BY(mu_);  // Maintain our own buffer
      size_t pos_ GUARDED_BY(
          mu_);  // Index into the buffer must be maintained between iters
      std::unique_ptr<io::RandomAccessInputStream> input_stream_
          GUARDED_BY(mu_);
      size_t current_file_index_ GUARDED_BY(mu_) = 0;
      std::unique_ptr<RandomAccessFile> file_
          GUARDED_BY(mu_);  // must outlive input_stream_
    };                      // class Iterator

    const std::vector<string> filenames_;
    const bool header_;
    const int64 buffer_size_;
    const DataTypeVector out_type_;
    const std::vector<PartialTensorShape> output_shapes_;
    const std::vector<Tensor> record_defaults_;
    const std::vector<int64> select_cols_;
    const bool use_quote_delim_;
    const char delim_;
    const string na_value_;
  };  // class Dataset

  DataTypeVector output_types_;
  std::vector<PartialTensorShape> output_shapes_;
};  // class CSVDatasetOp

// Register the kernel implementation for CSVDataset.
REGISTER_KERNEL_BUILDER(Name("CSVDataset").Device(DEVICE_CPU), CSVDatasetOp);

}  // namespace
}  // namespace tensorflow
