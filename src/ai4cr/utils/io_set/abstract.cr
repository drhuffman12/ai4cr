require "./file_type"

module Ai4cr
  module Utils
    module IoSet
      abstract class Abstract
        property file_path : String
        property raw = ""
        property ios = Array(Array(Float64)).new

        def initialize(@file_path : String, file_content_type : FileType)
          if file_content_type == FileType::Raw
            @raw = File.read(file_path)
            converted = convert_raw_to_ios(@raw)
            @ios = converted
          else
            contents = File.read(file_path)
            @ios = Array(Array(Float64)).from_json(contents)
            converted = convert_ios_to_raw(@ios)
            @raw = converted
          end
        end

        def convert_raw_to_ios(raw) : Array(Array(Float64))
          raise "Must be implemented in subclass"
          Array(Array(Float64)).new
        end

        def convert_ios_to_raw(ios) : String
          raise "Must be implemented in subclass"
          ""
        end
      end
    end
  end
end
