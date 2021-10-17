
echo "Copying files"
echo "-----------------------"
cp -R ../hw01/code/* ./
cp -R ../hw02/code/* ./
cp -R ../hw03/code/* ./
cp -R ../hw04/code/* ./
cp -R ../hw05/code/* ./

echo "Pushing to github"
echo "-----------------------"

git add .
git commit -m 'yet another commit3'
git push -u origin master
