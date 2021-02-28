require "./file_type"

module Ai4cr
  module Utils
    module IoSet
      abstract class Abstract
        property file_path : String
        property raw = ""
        property ios = Array(Array(Float64)).new

        def initialize(@file_path : String, file_content_type : FileType)
          case file_content_type
          when FileType::Raw
            @raw = File.read(file_path)
            @ios = convert_raw_to_ios(@raw)
          when FileType::Ios
            contents = File.read(file_path)
            @ios = Array(Array(Float64)).from_json(contents)
            @raw = convert_ios_to_raw(@ios)
          end
        end

        def convert_raw_to_ios(raw) : Array(Array(Float64))
          raise "Must be implemented in subclass"
        end

        def convert_ios_to_raw(ios) : String
          raise "Must be implemented in subclass"
        end

        def save_raw(to_file_path : String)
          File.write(to_file_path, raw)
        end

        def save_ios(to_file_path : String)
          File.write(to_file_path, ios.to_json)
        end
      end
    end
  end
end
