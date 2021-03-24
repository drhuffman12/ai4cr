folder_from = "./spec_bench/support/neural_network/data/bible_utf/"
folder_to = "./spec_bench/support/neural_network/data/bible_ascii/"

file_names = [
  "eng-web_002_GEN_01_read.txt",
  "eng-web_002_GEN_chap1-2.txt",
  "eng-web_002_GEN_chap1-4.txt",
  "eng-web_002_GEN_chap1-8.txt"
]

file_names.each do |file_name|
  utf_file_path = folder_from + file_name
  ascii_file_path = folder_to + file_name

  # utf_file = File.open(utf_file_path)
  ascii_file = File.open(ascii_file_path, mode="w")

  # ascii_file.set_encoding("WINDOWS-1252", nil)
  ascii_file.set_encoding("ASCII", nil)

  File.read_lines(filename: utf_file_path, encoding: "UTFinvalid: nil).each do |line|
    ascii_file.print line
  end
end
