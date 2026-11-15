import { test, expect } from '@playwright/test';

test.beforeEach(async ({ page }) => {
  const errors = [];
  page.on('pageerror', (e) => errors.push(e.message));
  page.errors = errors;
  await page.goto('/');
  await page.locator('#console[data-ready]').waitFor({ state: 'attached' });
  await page.locator('[data-whatif][data-ready]').waitFor({ state: 'attached' });
});

test('hero names the category and links to the replay', async ({ page }) => {
  await expect(page.getByRole('heading', { level: 1 })).toHaveText(/Fraud screening for small online stores/);
  await expect(page.locator('.hero').getByRole('link', { name: 'Run the live replay' })).toHaveAttribute('href', '#replay');
  expect(page.errors).toEqual([]);
});

test('replay streams orders and updates totals', async ({ page }) => {
  const orders = page.locator('[data-k="n"]');
  const before = Number((await orders.textContent()).replace(/,/g, ''));
  expect(before).toBeGreaterThan(0);
  await page.getByRole('button', { name: 'Add 100 orders' }).click();
  await expect(orders).toHaveText((before + 100).toLocaleString('en-US'));
  await page.getByRole('button', { name: 'Run replay' }).click();
  await expect(page.getByRole('button', { name: 'Pause replay' })).toHaveAttribute('aria-pressed', 'true');
  await expect.poll(async () => Number((await orders.textContent()).replace(/,/g, '')), { timeout: 15000 }).toBeGreaterThan(before + 100);
  await page.getByRole('button', { name: 'Pause replay' }).click();
});

test('picking an order explains its decision', async ({ page }) => {
  await page.locator('.rp-row').first().click();
  const pane = page.locator('[data-case]');
  await expect(pane.getByText('Expected cost of each action')).toBeVisible();
  await expect(pane.locator('.mini-cost__row')).toHaveCount(3);
  await expect(pane.getByText('Simulator truth')).toBeVisible();
});

test('reset returns the replay to its starting state', async ({ page }) => {
  const orders = page.locator('[data-k="n"]');
  const start = await orders.textContent();
  await page.getByRole('button', { name: 'Add 100 orders' }).click();
  await page.getByRole('button', { name: 'Reset replay' }).click();
  await expect(orders).toHaveText(start);
});

test('what-if re-optimises when a cost changes', async ({ page }) => {
  const save = page.locator('[data-w="save"]');
  await page.locator('#whatif').scrollIntoViewIfNeeded();
  const before = await save.textContent();
  await page.locator('#s-friction').fill('5');
  await expect(page.locator('[data-o="friction"]')).toHaveText('$5.00');
  await expect(save).not.toHaveText(before);
});

test('every in-page link lands on a section', async ({ page }) => {
  const hrefs = await page.locator('a[href^="#"]').evaluateAll((as) => [...new Set(as.map((a) => a.getAttribute('href')))]);
  for (const h of hrefs) expect(await page.locator(h).count(), h).toBe(1);
});

test('no horizontal scroll', async ({ page }) => {
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
  expect(overflow).toBeLessThanOrEqual(0);
});

test('run overview shows the cost chart and returns after a case', async ({ page }) => {
  const pane = page.locator('[data-case]');
  await expect(pane.locator('.ov-chart')).toBeVisible();
  await expect(pane.getByText('Fraud stopped by pattern')).toBeVisible();
  await page.locator('.rp-row').first().click();
  await pane.getByRole('button', { name: /Back to run overview/ }).click();
  await expect(pane.locator('.ov-chart')).toBeVisible();
});

test('hero shows the model visualisation and its pause control works', async ({ page }) => {
  await page.locator('#hero-viz[data-ready]').waitFor({ state: 'attached' });
  await expect(page.getByRole('img', { name: /gradient-boosted tree models/ })).toBeVisible();
  const pause = page.getByRole('button', { name: 'Pause model animation' });
  await pause.click();
  await expect(page.getByRole('button', { name: 'Play model animation' })).toHaveAttribute('aria-pressed', 'true');
});

test('cost breakdown parts add up to the totals', async ({ page }) => {
  const money = async (sel) => Number((await page.locator(sel).allTextContents()).join('|').replace(/[$,]/g, '').split('|').reduce((a, b) => a + Number(b), 0));
  expect(await money('.cb-row:not(.cb-row--sen) .cb-legend b')).toBe(77219);
  expect(await money('.cb-row--sen .cb-legend b')).toBe(14776);
});
