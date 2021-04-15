require "json"

puts "*"*20
f = 0.0
p!(f = Float64::NAN)
p! f
j = f.to_json
p! j
f2 = Float64.from_json(j)
p! f2

# This outputs:
# ...
# ********************
# f = Float64::NAN # => NaN
# f # => NaN
# Unhandled exception: NaN not allowed in JSON (JSON::Error)
#   from /usr/share/crystal/src/json/builder.cr:90:9 in 'number'
#   from /usr/share/crystal/src/json/to_json.cr:55:5 in 'to_json'
#   from /usr/share/crystal/src/json/to_json.cr:10:7 in 'to_json'
#   from /usr/share/crystal/src/json/to_json.cr:4:7 in 'to_json'
#   from examples/json_float_nan.cr:7:7 in '__crystal_main'
#   from /usr/share/crystal/src/crystal/main.cr:110:5 in 'main_user_code'
#   from /usr/share/crystal/src/crystal/main.cr:96:7 in 'main'
#   from /usr/share/crystal/src/crystal/main.cr:119:3 in 'main'
#   from __libc_start_main
#   from _start
#   from ???
