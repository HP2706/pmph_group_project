import "lib/github.com/diku-dk/sorts/radix_sort"


let sort_u8 = radix_sort u8.num_bits u8.get_bit

-- this is so ugly 
let main (arr: []u8) : []u8 = 
  sort_u8 arr