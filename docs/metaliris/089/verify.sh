#!/bin/zsh
set -eux
qt="${1:?provide native Qt build}"
core="${2:?provide native core build}"
output="${3:?provide scratch verification directory}"
root="${0:a:h:h:h:h}"
cmake --build "$qt" --target citra_meta --parallel 10
if [[ -n "${4:-}" ]]; then
    source "$4"
fi
python3 "$root/docs/metaliris/083/verify.py" "$qt" "$core" "$output/graphics"
python3 "$root/docs/metaliris/085/verify.py" "$qt" "$core" "$output/config"
ctest --test-dir "$core" -C Release --output-on-failure -j 4
