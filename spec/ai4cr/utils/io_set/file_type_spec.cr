require "./../../../spectator_helper"

Spectator.describe Ai4cr::Utils::IoSet::FileText do
  let(file_type_raw) { Ai4cr::Utils::IoSet::FileType::Raw }
  let(file_type_io_formatted) { Ai4cr::Utils::IoSet::FileType::IoFormatted }

  describe "FileType" do
    context "Raw" do
      it "#to_s" do
        expect(file_type_raw.to_s).to eq("Raw")
      end
      
      it "#to_json" do
        expect(file_type_raw.to_json).to eq("0")
      end
    end

    context "IoFormatted" do
      it "#to_s" do
        expect(file_type_io_formatted.to_s).to eq("IoFormatted")
      end
      
      it "#to_json" do
        expect(file_type_io_formatted.to_json).to eq("1")
      end
    end
  end
end
