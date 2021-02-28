require "./../../../spectator_helper"

Spectator.describe Ai4cr::Utils::IoData::FileText do
  let(file_type_raw) { Ai4cr::Utils::IoData::FileType::Raw }
  let(file_type_io_formatted) { Ai4cr::Utils::IoData::FileType::Iod }

  describe "FileType" do
    context "Raw" do
      it "#to_s" do
        expect(file_type_raw.to_s).to eq("Raw")
      end

      it "#to_json" do
        expect(file_type_raw.to_json).to eq("0")
      end
    end

    context "Iod" do
      it "#to_s" do
        expect(file_type_io_formatted.to_s).to eq("Iod")
      end

      it "#to_json" do
        expect(file_type_io_formatted.to_json).to eq("1")
      end
    end
  end
end
