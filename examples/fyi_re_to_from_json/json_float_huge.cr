require "json"
# struct Float64
#   include ::JSON::Serializable
# end

# Test what max values can be to/from-jsonified
(0..309).each do |exp|
  puts "*"*20
  p! exp
  f = 0.0
  p!(f = 10.0**(exp))
  p! f
  j = f.to_json
  p! j
  f2 = Float64.from_json(j)
  p! f2
end

# This outputs:
# ...
# ********************
# exp # => 308
# f = 10.0 ** (exp) # => 1.0000000000000006e+308
# f # => 1.0000000000000006e+308
# j # => "1.0000000000000006e+308"
# f2 # => 1.0000000000000012e+308
# ********************
# exp # => 309
# f = 10.0 ** (exp) # => Infinity
# f # => Infinity
# Unhandled exception: Infinity not allowed in JSON (JSON::Error)
#   from /usr/share/crystal/src/json/builder.cr:92:9 in 'number'
#   from /usr/share/crystal/src/json/to_json.cr:55:5 in 'to_json'
#   from /usr/share/crystal/src/json/to_json.cr:10:7 in 'to_json'
#   from /usr/share/crystal/src/json/to_json.cr:4:7 in 'to_json'
#   from examples/json_float_huge.cr:13:7 in '__crystal_main'
#   from /usr/share/crystal/src/crystal/main.cr:110:5 in 'main_user_code'
#   from /usr/share/crystal/src/crystal/main.cr:96:7 in 'main'
#   from /usr/share/crystal/src/crystal/main.cr:119:3 in 'main'
#   from __libc_start_main
#   from _start
#   from ???
