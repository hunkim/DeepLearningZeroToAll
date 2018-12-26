# ‘모두가 만드는 모두를 위한 딥러닝’ 참여 방법!! (Contribution)

## Precheck steps : 사전확인

*  작업을 시작하기 전에 먼저 이슈를 남겨 두세요. 왜냐면
  * 당신이 무엇을 하고 있는지 사람들에게 알리는 데 도움이 됩니다.
  * 문제는 이 Repo와 무관할 수 있습니다.
  * 그런 방식으로 코드를 유지하는 게 우리의 의도일 수도 있습니다. ([KISS](https://en.wikipedia.org/wiki/KISS_principle))
* 여러분은 git을 어떻게 사용하는지 알아야합니다.
  * 그렇지 않다면, "git 사용 방법"을 검색한 후, 무언가를 하기 전에 그것들을 읽어 보세요. 개발자로서 살아남기 위해서는 필수적인 기술입니다.
  * Git tutorial을 참고하세요 [git tutorial](https://try.github.io/levels/1/challenges/1)

## Contribution guidelines

This document will guide you through the contribution process.

### Step 1: Fork

Fork the project [on GitHub](https://github.com/hunkim/DeepLearningZeroToAll.git) by pressing the "fork" button.
This step will copy this repository to your account so that you can start working on.

### Step 2: Download to a local computer


```bash
$ git clone https://github.com/`YOUR_GITHUB_NAME`/DeepLearningZeroToAll.git
$ cd DeepLearningZeroToAll
```
### Step 3: Setup an upstream

It's always a good idea to set up a link to this repo so that you can pull easily if there were changes

```bash
$ git remote add upstream https://github.com/hunkim/DeepLearningZeroToAll.git
```

If there were updates in this repository, you can now keep your local copy and your repository updated

```bash
$ git pull upstream master && git push origin master
```

### Step 4: Make a branch

You don't want to directly modify the master branch because the master branch keeps changing by merging PRs, etc.

Also remember to give a meaningful name!

Example: 
```bash
$ git checkout -b hotfix/lab10 -t origin/master
```

After making a new branch, feel free to modify the codes now!

**Note: don't get tempted to fix other things that are not related to your issue.**
Your commit should be in logical blocks! If it's a different problem, you have to create a separate issue.


### Step 5: Commit

If you have not set up, please set up your email/username 
```bash
$ git config --global user.name "Sung Kim"
$ git config --global user.email "sungkim@email.com"
```

then commit:
```bash
$ git add my/changed/files
$ git commit
```

Notes
* Write a clear commit message!
* Example:
```text
Short (50 chars or less) summary of changes

More detailed explanatory text, if necessary.  Wrap it to about 72
characters or so.  In some contexts, the first line is treated as the
subject of an email and the rest of the text as the body.  The blank
line separating the summary from the body is critical (unless you omit
the body entirely); tools like rebase can get confused if you run the
two together.

Further paragraphs come after blank lines.

  - Bullet points are okay, too

  - Typically a hyphen or asterisk is used for the bullet, preceded by a
    single space, with blank lines in between, but conventions vary here
```

### Step 6: (Optional) Rebase your branch

If your fix is taking longer than usual, it's likely that your repo is outdated.  
Sync your repo to the latest:
```bash
$ git fetch upstream
$ git rebase upstream/master
```

### Step 7: Push

Before pushing to YOUR REPO, make sure you run `autopep8`!

Please follow PEP8 styles.  The only exception is `E501`(max-line-char limit)  
**Remember: Readability > everything**

Example:

```bash
$ autopep8 . -r -i --ignore E501
$ git push -u origin hotfix/lab10
```


### Step 8: Creating the PR
Now, if you open a browser and open this repo.  
You will see the big green button saying "compare & pull request."

* Please ensure you write a good title.
* Don't just write filenames you modified.
* **Explain what you did and why you did.**
* Add a relevant issue number as well.


Congratulations ! Your PR will be reviewed by collaborators.  
Please check your PR pass the CI test as well.
