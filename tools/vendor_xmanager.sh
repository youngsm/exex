#!/bin/bash

rm -rf exex/_vendor/xmanager
mkdir -p exex/_vendor/xmanager

cp -r third_party/xmanager/xmanager exex/_vendor/
rm -r exex/_vendor/xmanager/{docker,generated,contrib,cloud,cli,vizier,xm_local}
# this test uses xm_local
rm exex/_vendor/xmanager/xm/packagables_test.py

find exex/_vendor/xmanager -name '*.py' -exec \
  sed -i 's/^from xmanager/from exex\._vendor\.xmanager/' {} \;
