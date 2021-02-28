require "./../../../spectator_helper"

Spectator.describe Ai4cr::Utils::IoData::FileText do
  let(temp_folder) { "spec/tmp" }

  before_each do
    Dir.mkdir_p(temp_folder)
  end

  let(file_type_raw) { Ai4cr::Utils::IoData::FileType::Raw }
  let(file_type_iod) { Ai4cr::Utils::IoData::FileType::Iod }

  let(file_path) { "./spec_bench/support/neural_network/data/eng-web_002_GEN_01_read.txt" }
  let(io_set_text_file) { Ai4cr::Utils::IoData::TextFile.new(file_path, file_type_raw) }

  let(raw) { io_set_text_file.raw }
  let(iod) { io_set_text_file.iod }

  let(start_expected_10_chars) { "﻿The First" }
  let(end_expected_10_chars) { "sixth day. \n" }

  let(start_expected_3_iod) {
    [
      [
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      ],
      [
        0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      ],
      [
        0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      ],
    ]
  }
  let(end_expected_3_iod) {
    [
      [
        0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      ],
      [
        0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      ],
      [
        0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      ],
    ]
  }

  let(start_expected_3_chars) { "﻿Th" }
  let(end_expected_3_chars) { ". \n" }

  describe "#initialize" do
    describe "when given raw text data" do
      describe "assigns" do
        describe "raw, which" do
          it "starts as expected" do
            expect(raw[0..9]).to eq(start_expected_10_chars)
          end
          it "ends as expected" do
            expect(raw[-12..-1]).to eq(end_expected_10_chars)
          end
        end

        describe "iod, which" do
          it "starts as expected" do
            expect(iod[0..2]).to eq(start_expected_3_iod)
          end
          it "ends as expected" do
            expect(iod[-3..-1]).to eq(end_expected_3_iod)
          end
        end
      end
    end

    describe "when given iod data" do
      let(temp_file_path) { "#{temp_folder}/temp_text_file_spec.#{Time.utc}.#{rand(1000)}.json" }

      let(iod_file) do
        io_set_text_file.save_iod(temp_file_path)
        Ai4cr::Utils::IoData::TextFile.new(temp_file_path, file_type_iod)
      end

      describe "assigns" do
        describe "raw, which" do
          it "starts as expected" do
            expect(raw[0..9]).to eq(start_expected_10_chars)
          end
          it "ends as expected" do
            expect(raw[-12..-1]).to eq(end_expected_10_chars)
          end
        end

        describe "iod, which" do
          it "starts as expected" do
            expect(iod[0..2]).to eq(start_expected_3_iod)
          end
          it "ends as expected" do
            expect(iod[-3..-1]).to eq(end_expected_3_iod)
          end
        end
      end
    end
  end

  describe "#convert_raw_to_iod" do
    context "when given the first 3 raw charaters" do
      it "correctly converts to iod" do
        converted = io_set_text_file.convert_raw_to_iod(start_expected_3_chars)
        expect(converted).to eq(start_expected_3_iod)
      end
    end

    context "when given the last 3 raw charaters" do
      it "correctly converts to iod" do
        converted = io_set_text_file.convert_raw_to_iod(end_expected_3_chars)
        expect(converted).to eq(end_expected_3_iod)
      end
    end
  end

  describe "#convert_iod_to_raw" do
    context "when given the first 3 iod charaters" do
      it "correctly converts to raw" do
        converted = io_set_text_file.convert_iod_to_raw(start_expected_3_iod)
        expect(converted).to eq(start_expected_3_chars)
      end
    end

    context "when given the last 3 iod charaters" do
      it "correctly converts to raw" do
        converted = io_set_text_file.convert_iod_to_raw(end_expected_3_iod)
        expect(converted).to eq(end_expected_3_chars)
      end
    end
  end

  describe "#bits_to_char" do
    let(bits) { end_expected_3_iod.first }
    let(char_expected) { end_expected_3_chars[0].to_s }

    it "correctly converts to char" do
      converted = io_set_text_file.bits_to_char(bits)
      expect(converted).to eq(char_expected)
    end
  end

  describe "#bytes_to_chars" do
    let(iod) { end_expected_3_iod }
    let(raw_expected) { end_expected_3_chars.split("") }

    it "correctly converts to raw text array" do
      converted = io_set_text_file.bytes_to_chars(iod)
      expect(converted).to eq(raw_expected)
    end
  end

  context "saving" do
    describe "#save_raw" do
      let(temp_file_path) { "#{temp_folder}/temp_text_file_spec.#{Time.utc.to_s("%Y-%m-%d_%H-%M-%S")}.#{rand(1000)}.txt" }

      it "correctly saves the raw contents" do
        io_set_text_file.save_raw(temp_file_path)

        contents = File.read(temp_file_path)

        expect(contents).to eq(io_set_text_file.raw)

        begin
          File.delete(temp_file_path)
        rescue
        end
      end
    end

    describe "#save_iod" do
      let(temp_file_path) { "#{temp_folder}/temp_text_file_spec.#{Time.utc}.#{rand(1000)}.json" }

      it "correctly saves the iod as json" do
        io_set_text_file.save_iod(temp_file_path)

        contents = File.read(temp_file_path)

        expect(contents).to eq(io_set_text_file.iod.to_json)

        begin
          File.delete(temp_file_path)
        rescue
        end
      end
    end
  end
end
