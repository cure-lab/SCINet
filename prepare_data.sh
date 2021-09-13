#!/usr/bin/env bash
pip install gdown

mkdir -p datasets
cd datasets

mkdir -p PEMS
cd PEMS

gdown "https://drive.google.com/uc?id=1UQ-Mz-AJLieia-woSMtx8aW6igxPDDxL"
gdown "https://drive.google.com/uc?id=1mjM4xCf2GKWM5w6sCGuiK-o-RbRAZGMF"
gdown "https://drive.google.com/uc?id=1tekQqMPFjtT0I4JyV-SgToLn4K0KxNdf"
gdown "https://drive.google.com/uc?id=1IqbRJYuvxIwuaK1jpXvzrsDa7LxoD9zL"

cd ..

mkdir -p financial
cd financial

gdown "https://drive.google.com/uc?id=1ttSg9i3bzTI77oVoUU-odi67_NVbzFBx"
gdown "https://drive.google.com/uc?id=1zSKR2tORND40BBheWgtoCwgx1pTNntXM"
gdown "https://drive.google.com/uc?id=1MGIl1Aqnek0rPoPyqgS_Wzo5eQgIjihh"
gdown "https://drive.google.com/uc?id=1bp9J5PeA4lbj1wPXa4oxDX-vXuG_UyNz"

cd ..

mkdir -p ETT-data
cd ETT-data
gdown "https://drive.google.com/uc?id=10D9h6dVrlXknwYgYdnct8OfIaCRzIQXD"
gdown "https://drive.google.com/uc?id=18S5BrHOLrgqmTba2pOLWNxldIT9hrEGd"
gdown "https://drive.google.com/uc?id=1bxBD_uN1Gt3Tyn8Vb71ciAYyIbL4sZl1"
cd ..


