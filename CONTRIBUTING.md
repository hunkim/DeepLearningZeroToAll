# Contributing to DeepLearningZeroToAll

## Precheck steps

* Before starting to work on something, **please leave an issue first.** Because
  * It helps to let people know what you are working on
  * A problem might have nothing to do with this repo
  * It could be our intention to keep the code in that way (for [KISS](https://en.wikipedia.org/wiki/KISS_principle))
* You should know how to use git
  * If not, please google "how to use git." and read through them before you do anything. It's a mandatory skill to survive as a developer
  * Try [git tutorial](https://try.github.io/levels/1/challenges/1)

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
